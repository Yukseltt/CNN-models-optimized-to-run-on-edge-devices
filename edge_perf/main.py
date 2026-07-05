#!/usr/bin/env python3
# main.py
#
# edge_perf'in TEK giris noktasi.
#
#   python edge_perf/main.py
#
# Calistigi platformu otomatik algilar (server / jetson / rpi5_hailo), 5 modelin
# 3 varyantini (normal / pruned / quantized) tek tek olcer, kaydeder ve SIRALAR.
#
# Platform basina varyant->precision matrisi config.PLATFORM_VARIANTS'tadir:
#   server      : torch CUDA/CPU baseline (normal fp32, pruned fp32, quantized=fp16 proxy)
#   jetson      : TensorRT (normal/pruned fp16, quantized int8)
#   rpi5_hailo  : HailoRT int8 (.hef); frcnn/rtdetr -> Pi5 CPU onnx fallback
#
# Cihaz backend'leri DERLENMIS artifact tuketir (weights/ icinde *.engine / *.hef /
# *.onnx). Once convert/ scriptleriyle uret. Eksikse o varyant atlanir (SKIP).
# Sunucu (torch) backend dogrudan .pt'den calisir -> H200'de simdi calistirilabilir.

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_core import (                      # noqa: E402
    latency_stats, BackgroundSampler, evaluate_map, save_results,
    derive_efficiency, list_images, peak_rss_mb,
)
import config                                 # noqa: E402
from backends import (                        # noqa: E402
    detect_platform, ServerTorchBackend, TrtYoloBackend, TrtOnnxBackend,
    HailoBackend, CpuOnnxBackend,
)


# =====================================================================
# Platforma ozel guc / sicaklik ornekleme
# =====================================================================

class JetsonPower:
    def __init__(self):
        self.jtop = None
        try:
            from jtop import jtop
            self._ctx = jtop(); self._ctx.start(); self.jtop = self._ctx
            print("[power] jtop aktif")
        except Exception as e:
            print(f"[power] jtop yok ({e})")

    def watts(self):
        if self.jtop is None:
            return None
        try:
            p = self.jtop.power
            tot = p.get("tot") or p.get("all")
            if isinstance(tot, dict):
                return tot.get("power", tot.get("avg", 0)) / 1000.0
        except Exception:
            return None
        return None

    def temp_c(self):
        if self.jtop is None:
            return None
        try:
            vals = []
            for v in self.jtop.temperature.values():
                tv = v.get("temp") if isinstance(v, dict) else v
                if isinstance(tv, (int, float)) and tv > 0:
                    vals.append(tv)
            return max(vals) if vals else None
        except Exception:
            return None

    def close(self):
        if self.jtop is not None:
            try:
                self._ctx.close()
            except Exception:
                pass


def _vcgencmd(args):
    try:
        return subprocess.check_output(["vcgencmd"] + args, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def pi5_board_watts():
    out = _vcgencmd(["pmic_read_adc"])
    if not out:
        return None
    cur, volt = {}, {}
    for line in out.splitlines():
        mc = re.search(r"(\S+)_A\s+current\(\d+\)=([\d.]+)A", line)
        mv = re.search(r"(\S+)_V\s+volt\(\d+\)=([\d.]+)V", line)
        if mc:
            cur[mc.group(1)] = float(mc.group(2))
        if mv:
            volt[mv.group(1)] = float(mv.group(2))
    total = sum(cur[r] * volt[r] for r in cur if r in volt)
    return round(total, 3) if total > 0 else None


def pi5_temp_c():
    out = _vcgencmd(["measure_temp"])
    if out:
        m = re.search(r"temp=([\d.]+)", out)
        if m:
            return float(m.group(1))
    return None


# =====================================================================
# Backend kurucu
# =====================================================================

def build_backend(model, structure, precision, backend_kind, args):
    mtype = model["type"]

    if backend_kind == "torch":
        if precision == "int8":
            return None, "int8 sunucu torch'ta desteklenmez (jetson/rpi'de olcun)"
        pt = config.resolve_source_pt(model, structure)
        if pt is None:
            return None, f"kaynak .pt bulunamadi ({structure})"
        b = ServerTorchBackend(mtype, pt, model["imgsz"], precision,
                               conf=args.conf, device=args.device)
        return b, str(pt)

    if backend_kind == "trt":
        art = config.resolve_device_artifact(model, structure, "trt", precision)
        if not art.exists():
            return None, f"TRT artifact yok: {art.name} (once convert/ ile uret)"
        if mtype == "yolo":
            return TrtYoloBackend(art, model["imgsz"], conf=args.conf), str(art)
        return TrtOnnxBackend(art, model["imgsz"], decode=mtype,
                              precision=precision, conf=args.conf), str(art)

    if backend_kind == "hailo":
        if mtype == "yolo":
            art = config.resolve_device_artifact(model, structure, "hailo")
            if not art.exists():
                return None, f"HEF yok: {art.name} (Hailo DFC ile uret)"
            return HailoBackend(art, model["imgsz"], conf=args.conf), str(art)
        # frcnn/rtdetr Hailo'da zor -> Pi5 CPU onnx fallback
        art = config.resolve_device_artifact(model, structure, "trt")  # .onnx
        if not art.exists():
            return None, f"ONNX yok: {art.name} (Pi5 CPU fallback icin gerekli)"
        return CpuOnnxBackend(art, model["imgsz"], decode=mtype, conf=args.conf), str(art)

    if backend_kind == "rpi":
        # RPi5: int8 ve YOLO icin .hef (Hailo-8L) varsa onu kullan; yoksa
        # onnxruntime-CPU'ya dus (base/pruned=fp32, quant_fp16=fp16, quant_int8=int8).
        if mtype == "yolo" and precision == "int8":
            hef = config.resolve_device_artifact(model, structure, "hailo")
            if hef.exists():
                return HailoBackend(hef, model["imgsz"], conf=args.conf), str(hef)
        onnx = config.resolve_device_artifact(model, structure, "rpi", precision)
        if onnx.exists():
            if mtype == "yolo":
                # ultralytics .onnx'i onnxruntime-CPU ile kosar (decode+NMS dahil).
                return TrtYoloBackend(onnx, model["imgsz"], conf=args.conf), str(onnx)
            return CpuOnnxBackend(onnx, model["imgsz"], decode=mtype, conf=args.conf), str(onnx)
        # ONNX yoksa (orn frcnn export'u kirilgan): torch-CPU yedegi (.pt'den).
        # Sadece fp32; fp16/int8 onnx gerektirir.
        if mtype != "yolo" and precision == "fp32":
            pt = config.resolve_source_pt(model, structure)
            if pt is not None:
                b = ServerTorchBackend(mtype, pt, model["imgsz"], "fp32",
                                       conf=args.conf, device="cpu")
                return b, str(pt)
        return None, (f"ONNX yok: {onnx.name} (export_onnx.py+quantize_onnx.py ile uret; "
                      f"frcnn icin torch-CPU yedegi yalniz fp32)")

    return None, f"bilinmeyen backend: {backend_kind}"


# =====================================================================
# Tek (model x varyant) olcumu
# =====================================================================

def bench_one(model, variant_tuple, platform, args, power):
    variant, structure, precision, backend_kind = variant_tuple
    name = f"{model['key']}_{variant}"
    print(f"\n{'-' * 70}\n[{name}] type={model['type']} structure={structure} "
          f"precision={precision} backend={backend_kind}")

    try:
        backend, artifact = build_backend(model, structure, precision, backend_kind, args)
    except Exception as e:
        print(f"[SKIP] {name}: backend kurulamadi ({e})")
        return None
    if backend is None:
        print(f"[SKIP] {name}: {artifact}")
        return None

    import cv2
    imgs = list_images(config.TEST_IMAGES)
    if not imgs:
        print(f"[SKIP] {name}: test goruntusu yok -> {config.TEST_IMAGES}")
        backend.close(); return None
    sample = cv2.imread(str(imgs[0]))

    # Warmup
    for _ in range(args.warmup):
        backend.infer(sample)

    # Olcum + (cihazda) guc/sicaklik ornekleme
    samplers = []
    if platform == "jetson" and power is not None:
        samplers = [BackgroundSampler(power.watts, 0.05, "power_w"),
                    BackgroundSampler(power.temp_c, 0.2, "temp_c")]
    elif platform == "rpi5_hailo":
        samplers = [BackgroundSampler(pi5_board_watts, 0.1, "power_w"),
                    BackgroundSampler(pi5_temp_c, 0.2, "temp_c")]
        if hasattr(backend, "chip_watts"):
            samplers.append(BackgroundSampler(backend.chip_watts, 0.1, "hailo_power_w"))
    for s in samplers:
        s.start()

    times = []
    for _ in range(args.measure):
        t0 = time.perf_counter()
        backend.infer(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    for s in samplers:
        s.stop()

    art_path = Path(artifact)
    m = {"name": name, "model": model["key"], "variant": variant,
         "type": model["type"], "structure": structure, "precision": precision,
         "backend": backend_kind, "imgsz": model["imgsz"], "artifact": artifact}
    m.update(latency_stats(times))
    for s in samplers:
        m.update(s.summary())

    # Param: torch gercek hesaplar; cihazda nominal config.
    m["params_m"] = (round(backend.params_m, 3) if getattr(backend, "params_m", None)
                     else (model.get("params_m_pruned") if structure == "pruned"
                           else model.get("params_m")))
    rss = peak_rss_mb()
    if rss:
        m["peak_rss_mb"] = round(rss, 1)
    if art_path.exists():
        m["disk_mb"] = round(art_path.stat().st_size / 1e6, 2)

    # mAP
    if not args.no_map:
        print(f"[{name}] mAP hesaplaniyor ({args.map_images or 'tum'} goruntu)...")
        try:
            mp = evaluate_map(backend.infer, config.TEST_GT, config.TEST_IMAGES,
                              max_images=args.map_images, class_names=config.CLASS_NAMES,
                              model_class_names=None)   # pozisyon-tabanli (bu proje)
            m["map50"] = mp["map50"]; m["map5095"] = mp["map5095"]
            m["per_class_ap50"] = mp["per_class"]
        except Exception as e:
            print(f"[{name}] mAP atlandi: {e}")

    m.update(derive_efficiency(m))
    backend.close()
    print(f"[{name}] FPS={m.get('fps_mean')} lat={m.get('latency_mean_ms')}ms "
          f"mAP@.5={m.get('map50')} params={m.get('params_m')}M W={m.get('power_w_avg')}")
    return m


# =====================================================================
# Sistem bilgisi
# =====================================================================

def system_info(platform):
    info = {"platform": platform}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if platform == "rpi5_hailo":
        try:
            import hailo_platform
            info["hailort"] = getattr(hailo_platform, "__version__", "?")
        except Exception:
            info["hailort"] = "yok"
    return info


def main():
    ap = argparse.ArgumentParser(description="edge_perf — tek main, tum modeller x varyant")
    ap.add_argument("--platform", choices=["auto", "server", "jetson", "rpi5_hailo"],
                    default="auto", help="varsayilan: otomatik algila")
    ap.add_argument("--models", default="all",
                    help="virgulle model key listesi (orn yolov8n,rtdetr_v2_r18vd) veya 'all'")
    ap.add_argument("--variants", default="all",
                    help="virgulle varyant listesi (normal,pruned,quantized) veya 'all'")
    ap.add_argument("--device", default=None, help="torch cihazi (cuda:0 / cpu); server icin")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="tespit conf esigi (mAP fideligi icin dusuk; COCO standardi "
                         "~0.001). RT-DETR/DETR ailesi dusuk-kalibre skor uretir, "
                         "yuksek esik mAP'i sifirlar.")
    ap.add_argument("--warmup", type=int, default=config.WARMUP_RUNS)
    ap.add_argument("--measure", type=int, default=config.MEASURE_RUNS)
    ap.add_argument("--map-images", type=int, default=config.MAP_MAX_IMAGES,
                    dest="map_images", help="mAP icin maksimum goruntu (None=tum)")
    ap.add_argument("--no-map", action="store_true", help="mAP'i atla (sadece hiz)")
    ap.add_argument("--smoke", action="store_true",
                    help="hizli duman testi: 5 warmup, 20 olcum, 50 mAP goruntu")
    args = ap.parse_args()

    if args.smoke:
        args.warmup, args.measure = 5, 20
        if not args.map_images:
            args.map_images = 50

    platform = detect_platform() if args.platform == "auto" else args.platform
    variants = config.PLATFORM_VARIANTS[platform]

    sel_models = (config.MODELS if args.models == "all"
                  else [m for m in config.MODELS if m["key"] in args.models.split(",")])
    if args.variants != "all":
        want = set(args.variants.split(","))
        variants = [v for v in variants if v[0] in want]

    print("=" * 70)
    print(f"EDGE PERF — platform={platform}  modeller={len(sel_models)}  "
          f"varyant={len(variants)}  (toplam {len(sel_models) * len(variants)} kosum)")
    print("=" * 70)

    power = JetsonPower() if platform == "jetson" else None
    results = {"device_info": system_info(platform), "platform": platform,
               "warmup": args.warmup, "measure": args.measure,
               "map_images": args.map_images, "models": []}

    for model in sel_models:
        for vt in variants:
            r = bench_one(model, vt, platform, args, power)
            if r:
                results["models"].append(r)

    if power is not None:
        power.close()

    if not results["models"]:
        print("\n[!] Hicbir sonuc uretilmedi. Cihaz artifact'lari (engine/hef/onnx) "
              "eksik olabilir; sunucuda --platform server ile torch baseline dene.")
        return

    save_results(results, str(config.RESULTS_DIR), f"edge_perf_{platform}")


if __name__ == "__main__":
    main()

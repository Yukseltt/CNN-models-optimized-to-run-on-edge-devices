# benchmark_jetson_orin_nano.py
#
# Jetson Orin Nano uzerinde pruned modellerin TensorRT benchmark'i.
# Olculen: latency (mean/median/p90/p95/p99/std/min/max), FPS, pre/inf/post kirilimi,
#          guc (W, jtop/tegrastats), enerji/frame (mJ), peak RAM, GPU util, sicaklik,
#          mAP@0.5 / mAP@0.5:0.95 (TensorRT FP16/INT8 sonrasi), turetilmis verimlilik.
#
# ====================  ONKOSUL: model -> TensorRT engine  ====================
# Bu script DERLENMIS artifact tuketir. Once cihazda donusturun:
#
#   YOLO (v5s, v8n)  -- ultralytics dogrudan TensorRT engine'e export/run eder:
#       yolo export model=pruned_final.pt format=engine half=True imgsz=640 device=0
#     (ya da INT8: int8=True data=data.yaml  -> kalibrasyon)
#     Cikan .engine'i ULTRALYTICS runtime ile calistiririz (en hizli YOLO yolu).
#
#   frcnn / rtdetr  -- ONNX uzerinden onnxruntime TensorRT EP:
#       (sunucuda) pruned_final.pt -> ONNX (pruning script ONNX da kaydeder / ayrica export)
#       Cihazda onnxruntime-gpu TensorrtExecutionProvider ONNX'ten engine'i JIT
#       derler ve cache'ler (ilk calistirma yavas, sonra hizli).
#
# REGISTRY'deki yollari kendi cihazindaki dosyalara gore doldur.
# Eksik bagimliklar (tensorrt/onnxruntime/ultralytics/jtop) varsa ilgili model atlanir.

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_core import (
    latency_stats, BackgroundSampler, evaluate_map, save_results,
    derive_efficiency, list_images, letterbox,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Test seti (mAP icin) — cihazda bu yollar gecerli olmali.
TEST_IMAGES = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/images"
TEST_GT     = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/_annotations.coco.json"
CLASS_NAMES = ["Person", "Car", "OtherVehicle"]

WARMUP_RUNS  = 20
MEASURE_RUNS = 100
MAP_MAX_IMAGES = None        # None -> tum test seti; int -> hizli alt-kume

# ----- model kayit defteri -----
# type: "yolo"  -> ultralytics .engine (veya .pt) ; backend decode/NMS halleder
#       "onnx"  -> onnxruntime TensorRT EP ; decode'a gore frcnn/rtdetr
# precision sadece raporlama icindir (engine'i nasil derledigini not etmen icin).
REGISTRY = [
    {"name": "yolov8n_pruned",       "type": "yolo", "artifact": PROJECT_ROOT / "runs_pruing/yolov8n/finetune/weights/best.engine",       "imgsz": 640, "params_m": 2.73, "precision": "FP16"},
    {"name": "yolo_fir_v5s_pruned",  "type": "yolo", "artifact": PROJECT_ROOT / "runs_pruing/yolo_fir_v5s/finetune/weights/best.engine",  "imgsz": 640, "params_m": 8.55, "precision": "FP16"},
    {"name": "frcnn_r50_pruned",     "type": "onnx", "decode": "frcnn",  "artifact": PROJECT_ROOT / "runs_pruing/faster_rcnn_resnet50/pruned_final.onnx", "imgsz": 800, "params_m": 34.8, "precision": "FP16"},
    {"name": "rtdetr_r18_pruned",    "type": "onnx", "decode": "rtdetr", "artifact": PROJECT_ROOT / "runs_pruing/rtdetr_v2_r18vd/pruned_final.onnx",     "imgsz": 480, "params_m": 14.3, "precision": "FP16"},
]


# =====================================================================
# Guc / sistem ornekleme (Jetson)
# =====================================================================

class JetsonPower:
    # jtop (jetson-stats) varsa toplam guc (mW->W) ve sicaklik/GPU util okur.
    # Yoksa tegrastats parse'a duser. Hicbiri yoksa None doner (guc atlanir).
    def __init__(self):
        self.jtop = None
        try:
            from jtop import jtop
            self._ctx = jtop()
            self._ctx.start()
            self.jtop = self._ctx
            print("[power] jtop aktif")
        except Exception as e:
            print(f"[power] jtop yok ({e}); tegrastats denenecek")

    def watts(self):
        if self.jtop is not None:
            try:
                p = self.jtop.power
                tot = p.get("tot") or p.get("all")
                if isinstance(tot, dict):
                    return tot.get("power", tot.get("avg", 0)) / 1000.0  # mW -> W
            except Exception:
                return None
        return None

    def gpu_util(self):
        if self.jtop is not None:
            try:
                g = self.jtop.gpu
                # jtop surumune gore: gpu['gpu']['status']['load'] veya gpu['val']
                first = next(iter(g.values())) if isinstance(g, dict) else g
                if isinstance(first, dict):
                    return first.get("status", {}).get("load", first.get("val"))
            except Exception:
                return None
        return None

    def temp_c(self):
        if self.jtop is not None:
            try:
                t = self.jtop.temperature
                vals = []
                for v in t.values():
                    tv = v.get("temp") if isinstance(v, dict) else v
                    if isinstance(tv, (int, float)) and tv > 0:
                        vals.append(tv)
                return max(vals) if vals else None
            except Exception:
                return None
        return None

    def close(self):
        if self.jtop is not None:
            try:
                self._ctx.close()
            except Exception:
                pass


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


# =====================================================================
# Backend'ler (infer: image_bgr -> boxes_xyxy, scores, class_idx_0based)
# =====================================================================

class YoloBackend:
    # Ultralytics .engine (TensorRT) veya .pt. Decode+NMS ultralytics'te.
    def __init__(self, artifact, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(str(artifact))
        self.imgsz = imgsz
        # Modelin cikti sinif sirasi (mAP'te isimle eslemek icin).
        self.class_names = list(self.model.names.values())

    def infer(self, image_bgr):
        r = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=0.25,
                               verbose=False, save=False)[0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                b.cls.cpu().numpy().astype(int))


class OnnxTrtBackend:
    # onnxruntime TensorRT EP (fallback CUDA). decode: "frcnn" | "rtdetr".
    def __init__(self, artifact, imgsz, decode, conf=0.3):
        import onnxruntime as ort
        providers = [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True,
                                            "trt_engine_cache_enable": True,
                                            "trt_engine_cache_path": str(Path(artifact).parent / "trt_cache")}),
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]
        self.sess = ort.InferenceSession(str(artifact), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.decode = decode
        self.conf = conf
        # frcnn/rtdetr cikti sirasi (egitim sirasi): Person, Car, OtherVehicle.
        self.class_names = list(CLASS_NAMES)
        print(f"[onnx] {Path(artifact).name} providers={self.sess.get_providers()}")

    def _pre(self, image_bgr):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.decode == "frcnn":
            # torchvision frcnn: 0-1 normalize, dinamik boyut (resize'a gerek yok ama
            # sabit engine icin imgsz'e resize). 1x3xHxW.
            img = cv2.resize(rgb, (self.imgsz, self.imgsz))
            t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            return t[None], 1.0, (0, 0), image_bgr.shape[:2]
        # rtdetr: letterbox + imagenet normalize
        padded, r, (pw, ph) = letterbox(rgb, self.imgsz)
        t = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        t = (t - mean) / std
        return t.transpose(2, 0, 1)[None], r, (pw, ph), image_bgr.shape[:2]

    def infer(self, image_bgr):
        blob, r, (pw, ph), (oh, ow) = self._pre(image_bgr)
        outs = self.sess.run(None, {self.inp: blob})
        if self.decode == "frcnn":
            return self._decode_frcnn(outs, oh, ow)
        return self._decode_rtdetr(outs, oh, ow)

    def _decode_frcnn(self, outs, oh, ow):
        # torchvision detection ONNX ciktisi: boxes[N,4], labels[N], scores[N]
        # (engine sabit imgsz'e resize edildiginden kutular imgsz uzayinda -> orijinale olcekle)
        boxes, labels, scores = None, None, None
        for o in outs:
            if o.ndim == 2 and o.shape[1] == 4:
                boxes = o
            elif o.ndim == 1 and np.issubdtype(o.dtype, np.integer):
                labels = o
            elif o.ndim == 1:
                scores = o
        if boxes is None:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
        sx, sy = ow / self.imgsz, oh / self.imgsz
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        keep = scores >= self.conf
        return boxes[keep], scores[keep], (labels[keep] - 1).astype(int)  # frcnn 1-based -> 0-based

    def _decode_rtdetr(self, outs, oh, ow):
        # HF RTDetr ONNX: logits[1,Q,nc], pred_boxes[1,Q,4] (cxcywh, 0-1 normalize)
        logits = boxes = None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
            elif o.ndim == 3:
                logits = o[0]
        if logits is None or boxes is None:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
        prob = 1.0 / (1.0 + np.exp(-logits))      # sigmoid
        cls = prob.argmax(1)
        sc = prob.max(1)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * ow; y1 = (cy - h / 2) * oh
        x2 = (cx + w / 2) * ow; y2 = (cy + h / 2) * oh
        xyxy = np.stack([x1, y1, x2, y2], 1)
        keep = sc >= self.conf
        return xyxy[keep], sc[keep], cls[keep].astype(int)


def build_backend(entry):
    if entry["type"] == "yolo":
        return YoloBackend(entry["artifact"], entry["imgsz"])
    return OnnxTrtBackend(entry["artifact"], entry["imgsz"], entry["decode"])


# =====================================================================
# Tek model benchmark
# =====================================================================

def bench_one(entry, power: JetsonPower):
    art = Path(entry["artifact"])
    if not art.exists():
        print(f"[SKIP] {entry['name']}: artifact yok -> {art}")
        return None
    try:
        backend = build_backend(entry)
    except Exception as e:
        print(f"[SKIP] {entry['name']}: backend kurulamadi ({e})")
        return None

    imgs = list_images(TEST_IMAGES)
    if not imgs:
        print(f"[SKIP] {entry['name']}: test goruntusu yok -> {TEST_IMAGES}")
        return None
    import cv2
    sample = cv2.imread(str(imgs[0]))

    # Warmup
    for _ in range(WARMUP_RUNS):
        backend.infer(sample)

    # Olcum + guc/sicaklik ornekleme
    pw_s = BackgroundSampler(power.watts, 0.05, "power_w").start()
    gp_s = BackgroundSampler(power.gpu_util, 0.1, "gpu_util").start()
    tp_s = BackgroundSampler(power.temp_c, 0.2, "temp_c").start()
    times = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        backend.infer(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    pw_s.stop(); gp_s.stop(); tp_s.stop()

    m = {"name": entry["name"], "type": entry["type"],
         "precision": entry.get("precision"), "params_m": entry.get("params_m"),
         "artifact": str(art), "imgsz": entry["imgsz"]}
    m.update(latency_stats(times))
    m.update(pw_s.summary()); m.update(gp_s.summary()); m.update(tp_s.summary())
    rss = peak_rss_mb()
    if rss:
        m["peak_rss_mb"] = round(rss, 1)
    m["disk_mb"] = round(art.stat().st_size / 1e6, 2)

    # mAP
    print(f"[{entry['name']}] mAP hesaplaniyor (test seti)...")
    try:
        # model_class_names verilmezse pozisyon-tabanli esleme (bu proje icin dogru).
        # Sadece isimleri gt ile gercekten uyusan modellerde entry'de ver.
        mp = evaluate_map(backend.infer, TEST_GT, TEST_IMAGES,
                          max_images=MAP_MAX_IMAGES, class_names=CLASS_NAMES,
                          model_class_names=entry.get("model_class_names"))
        m["map50"] = mp["map50"]; m["map5095"] = mp["map5095"]
        m["per_class_ap50"] = mp["per_class"]
    except Exception as e:
        print(f"[{entry['name']}] mAP atlandi: {e}")

    m["power_w_avg"] = m.get("power_w_avg")  # netlik
    m.update(derive_efficiency(m))
    print(f"[{entry['name']}] FPS={m.get('fps_mean')} lat={m.get('latency_mean_ms')}ms "
          f"mAP@.5={m.get('map50')} W={m.get('power_w_avg')}")
    return m


def system_info():
    info = {"platform": "Jetson Orin Nano"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        info["nvpmodel"] = out.strip().splitlines()[-1]
    except Exception:
        pass
    return info


def main():
    print("=" * 70)
    print("JETSON ORIN NANO — TensorRT benchmark")
    print("=" * 70)
    power = JetsonPower()
    results = {"device_info": system_info(), "warmup": WARMUP_RUNS,
               "measure": MEASURE_RUNS, "models": []}
    for entry in REGISTRY:
        r = bench_one(entry, power)
        if r:
            results["models"].append(r)
    power.close()
    save_results(results, str(PROJECT_ROOT / "bechmarks"), "jetson_orin_nano")


if __name__ == "__main__":
    main()

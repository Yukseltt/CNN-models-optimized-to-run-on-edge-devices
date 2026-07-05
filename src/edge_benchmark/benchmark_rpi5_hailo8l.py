# benchmark_rpi5_hailo8l.py
#
# Raspberry Pi 5 + AI HAT (Hailo-8L, 13 TOPS) uzerinde pruned modellerin benchmark'i.
# Olculen: latency (mean/median/p90/p95/p99/std/min/max), FPS, guc (W, Pi5 PMIC +
#          opsiyonel Hailo chip), enerji/frame (mJ), peak RAM, sicaklik, throttling,
#          mAP@0.5 / mAP@0.5:0.95 (INT8 sonrasi!), turetilmis verimlilik.
#
# ====================  ONKOSUL: model -> Hailo .hef  ====================
# Hailo ONNX/PyTorch CALISTIRMAZ; modelin Hailo Dataflow Compiler ile INT8'e
# kuantize edilip .hef'e DERLENMESI gerekir (offline, kalibrasyon verisiyle):
#
#   1) pruned_final.pt -> ONNX  (sunucuda; YOLO icin ultralytics export onnx)
#   2) hailo parser onnx ...          -> .har        (Hailo DFC)
#   3) hailo optimize ... --calib ...  -> quantized .har (INT8, kalibrasyon img'leri)
#   4) hailo compiler ...              -> .hef         (Hailo-8L hedef)
#   (YOLO icin Hailo Model Zoo hazir .yaml/postprocess sunar; en kolay yol.)
#
# NOT: Hailo-8L'de pratikte YOLO ailesi temiz derlenir. frcnn (two-stage) ve
# rtdetr (transformer) Hailo'da cok zor/desteksiz olabilir -> o modeller icin
# REGISTRY'de "cpu_onnx" kullanip Pi5 CPU (onnxruntime) baseline'i olcebilirsin.
#
# Eksik bagimliklar (hailort/onnxruntime) varsa ilgili model atlanir.

import os
import re
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

TEST_IMAGES = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/images"
TEST_GT     = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/_annotations.coco.json"
CLASS_NAMES = ["Person", "Car", "OtherVehicle"]

WARMUP_RUNS  = 20
MEASURE_RUNS = 100
MAP_MAX_IMAGES = None

# ----- model kayit defteri -----
# backend: "hailo"     -> .hef, HailoRT (Hailo-8L NPU)
#          "cpu_onnx"  -> Pi5 CPU onnxruntime baseline (Hailo'ya derlenemeyenler icin)
# hailo_nms: True -> .hef icinde on-chip NMS post-process var (cikti = per-class detection);
#            False -> ham tensor cikar (decode'u burada yapman gerekir, modele ozel).
REGISTRY = [
    {"name": "yolov8n_pruned",      "backend": "hailo", "artifact": PROJECT_ROOT / "hailo/yolov8n_pruned.hef",      "imgsz": 640, "params_m": 2.73, "hailo_nms": True, "precision": "INT8"},
    {"name": "yolo_fir_v5s_pruned", "backend": "hailo", "artifact": PROJECT_ROOT / "hailo/yolo_fir_v5s_pruned.hef", "imgsz": 640, "params_m": 8.55, "hailo_nms": True, "precision": "INT8"},
    # frcnn/rtdetr Hailo'da zor -> Pi5 CPU baseline:
    {"name": "frcnn_r50_pruned_cpu", "backend": "cpu_onnx", "decode": "frcnn",  "artifact": PROJECT_ROOT / "runs_pruing/faster_rcnn_resnet50/pruned_final.onnx", "imgsz": 800, "params_m": 34.8, "precision": "FP32-CPU"},
    {"name": "rtdetr_r18_pruned_cpu","backend": "cpu_onnx", "decode": "rtdetr", "artifact": PROJECT_ROOT / "runs_pruing/rtdetr_v2_r18vd/pruned_final.onnx",     "imgsz": 480, "params_m": 14.3, "precision": "FP32-CPU"},
]


# =====================================================================
# Pi5 guc / sicaklik / throttle (vcgencmd)
# =====================================================================

def _vcgencmd(args):
    try:
        return subprocess.check_output(["vcgencmd"] + args, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def pi5_board_watts():
    # Pi5 PMIC rail'lerinden toplam guc tahmini: her rail icin V*I topla.
    # pmic_read_adc ciktisi: "<RAIL>_A current(..)=X.XXA" ve "<RAIL>_V volt(..)=Y.YYV".
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
    out = _vcgencmd(["measure_temp"])      # "temp=52.0'C"
    if out:
        m = re.search(r"temp=([\d.]+)", out)
        if m:
            return float(m.group(1))
    return None


def pi5_throttled():
    out = _vcgencmd(["get_throttled"])     # "throttled=0x0"
    if out:
        m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out)
        if m:
            return m.group(1)
    return None


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


# =====================================================================
# Backend: Hailo (.hef)
# =====================================================================

class HailoBackend:
    # HailoRT InferVStreams ile .hef calistirir. Girdi NHWC uint8 (Hailo std).
    # hailo_nms=True ise cikti on-chip NMS formatinda (per-class [n,5]:
    # ymin,xmin,ymax,xmax,score, NORMALIZE) -> xyxy piksele cevrilir.
    def __init__(self, artifact, imgsz, hailo_nms=True, conf=0.3):
        from hailo_platform import (VDevice, HEF, ConfigureParams,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams,
                                     FormatType)
        self.imgsz = imgsz
        self.hailo_nms = hailo_nms
        self.conf = conf
        # Hailo on-chip NMS sinif sirasi = HEF derleme sirasi. HEF'in egitim
        # data.yaml sirasiyla derlendigini varsayar; farkliysa REGISTRY'de
        # entry['model_class_names'] ile ez.
        self.class_names = list(CLASS_NAMES)

        self.hef = HEF(str(artifact))
        self.target = VDevice()
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, cfg)[0]
        self.ng_params = self.network_group.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_infos = self.hef.get_output_vstream_infos()
        self.in_vp = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.out_vp = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.InferVStreams = InferVStreams
        # Girdi sekli (H,W,C)
        self.in_h, self.in_w = self.in_info.shape[0], self.in_info.shape[1]
        # power measurement (Hailo-8/8L destekliyse)
        self._power_ok = False
        try:
            from hailo_platform import DvmTypes, PowerMeasurementTypes  # noqa
            self.target.set_power_measurement()  # default rail
            self.target.start_power_measurement()
            self._power_ok = True
        except Exception:
            pass
        # InferVStreams context'i acik tut
        self._pipe_ctx = self.InferVStreams(self.network_group, self.in_vp, self.out_vp)
        self.pipe = self._pipe_ctx.__enter__()
        self._act_ctx = self.network_group.activate(self.ng_params)
        self._act_ctx.__enter__()

    def chip_watts(self):
        if not self._power_ok:
            return None
        try:
            m = self.target.get_power_measurement()
            return float(getattr(m, "average_value", None) or m)
        except Exception:
            return None

    def _pre(self, image_bgr):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        padded, r, (pw, ph) = letterbox(rgb, (self.in_h, self.in_w))
        return padded.astype(np.uint8)[None], r, (pw, ph), image_bgr.shape[:2]

    def infer(self, image_bgr):
        blob, r, (pw, ph), (oh, ow) = self._pre(image_bgr)
        res = self.pipe.infer({self.in_info.name: blob})
        out = res[self.out_infos[0].name]
        if self.hailo_nms:
            return self._decode_nms(out, r, pw, ph, oh, ow)
        # ham cikti -> modele ozel decode gerekir (burada bos doner; doldur)
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)

    def _decode_nms(self, out, r, pw, ph, oh, ow):
        # Hailo on-chip NMS: out[0] -> liste; her sinif icin ndarray [n,5]
        # (ymin,xmin,ymax,xmax,score), NORMALIZE [0,1] (letterbox uzayinda).
        dets = out[0] if isinstance(out, (list, tuple, np.ndarray)) and len(out) else out
        boxes, scores, labels = [], [], []
        H = self.in_h; W = self.in_w
        for cls_idx, arr in enumerate(dets):
            if arr is None or len(arr) == 0:
                continue
            arr = np.asarray(arr)
            for d in arr:
                ymin, xmin, ymax, xmax, sc = d[:5]
                if sc < self.conf:
                    continue
                # normalize -> letterbox piksel -> pad cikar -> orijinale olcekle
                x1 = (xmin * W - pw) / r; y1 = (ymin * H - ph) / r
                x2 = (xmax * W - pw) / r; y2 = (ymax * H - ph) / r
                boxes.append([x1, y1, x2, y2]); scores.append(float(sc)); labels.append(cls_idx)
        if not boxes:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
        return np.array(boxes), np.array(scores), np.array(labels, int)

    def close(self):
        try:
            self._act_ctx.__exit__(None, None, None)
            self._pipe_ctx.__exit__(None, None, None)
            if self._power_ok:
                self.target.stop_power_measurement()
            self.target.release()
        except Exception:
            pass


# =====================================================================
# Backend: Pi5 CPU onnxruntime (baseline / Hailo'ya derlenemeyenler)
# =====================================================================

class CpuOnnxBackend:
    def __init__(self, artifact, imgsz, decode, conf=0.3):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, len(os.sched_getaffinity(0)))
        self.sess = ort.InferenceSession(str(artifact), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.decode = decode
        self.conf = conf
        self.class_names = list(CLASS_NAMES)   # frcnn/rtdetr: Person, Car, OtherVehicle

    def _pre(self, image_bgr):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.decode == "frcnn":
            img = cv2.resize(rgb, (self.imgsz, self.imgsz))
            t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            return t[None], image_bgr.shape[:2]
        padded, r, (pw, ph) = letterbox(rgb, self.imgsz)
        t = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        t = ((t - mean) / std).transpose(2, 0, 1)[None]
        return t, image_bgr.shape[:2]

    def infer(self, image_bgr):
        blob, (oh, ow) = self._pre(image_bgr)
        outs = self.sess.run(None, {self.inp: blob})
        if self.decode == "frcnn":
            boxes = labels = scores = None
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
            boxes = boxes.copy(); boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
            keep = scores >= self.conf
            return boxes[keep], scores[keep], (labels[keep] - 1).astype(int)
        # rtdetr
        logits = boxes = None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
            elif o.ndim == 3:
                logits = o[0]
        if logits is None or boxes is None:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
        prob = 1.0 / (1.0 + np.exp(-logits))
        cls = prob.argmax(1); sc = prob.max(1)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([(cx - w / 2) * ow, (cy - h / 2) * oh,
                         (cx + w / 2) * ow, (cy + h / 2) * oh], 1)
        keep = sc >= self.conf
        return xyxy[keep], sc[keep], cls[keep].astype(int)

    def close(self):
        pass


def build_backend(entry):
    if entry["backend"] == "hailo":
        return HailoBackend(entry["artifact"], entry["imgsz"], entry.get("hailo_nms", True))
    return CpuOnnxBackend(entry["artifact"], entry["imgsz"], entry["decode"])


# =====================================================================
# Tek model benchmark
# =====================================================================

def bench_one(entry):
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
        print(f"[SKIP] {entry['name']}: test goruntusu yok")
        backend.close(); return None
    import cv2
    sample = cv2.imread(str(imgs[0]))

    for _ in range(WARMUP_RUNS):
        backend.infer(sample)

    samplers = [
        BackgroundSampler(pi5_board_watts, 0.1, "power_w"),
        BackgroundSampler(pi5_temp_c, 0.2, "temp_c"),
    ]
    if hasattr(backend, "chip_watts"):
        samplers.append(BackgroundSampler(backend.chip_watts, 0.1, "hailo_power_w"))
    for s in samplers:
        s.start()
    throttle_before = pi5_throttled()
    times = []
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        backend.infer(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    for s in samplers:
        s.stop()
    throttle_after = pi5_throttled()

    m = {"name": entry["name"], "backend": entry["backend"],
         "precision": entry.get("precision"), "params_m": entry.get("params_m"),
         "artifact": str(art), "imgsz": entry["imgsz"],
         "throttled_before": throttle_before, "throttled_after": throttle_after}
    m.update(latency_stats(times))
    for s in samplers:
        m.update(s.summary())
    rss = peak_rss_mb()
    if rss:
        m["peak_rss_mb"] = round(rss, 1)
    m["disk_mb"] = round(art.stat().st_size / 1e6, 2)

    print(f"[{entry['name']}] mAP hesaplaniyor...")
    try:
        # model_class_names verilmezse pozisyon-tabanli esleme (bu proje icin dogru).
        mp = evaluate_map(backend.infer, TEST_GT, TEST_IMAGES,
                          max_images=MAP_MAX_IMAGES, class_names=CLASS_NAMES,
                          model_class_names=entry.get("model_class_names"))
        m["map50"] = mp["map50"]; m["map5095"] = mp["map5095"]
        m["per_class_ap50"] = mp["per_class"]
    except Exception as e:
        print(f"[{entry['name']}] mAP atlandi: {e}")

    m.update(derive_efficiency(m))
    backend.close()
    print(f"[{entry['name']}] FPS={m.get('fps_mean')} lat={m.get('latency_mean_ms')}ms "
          f"mAP@.5={m.get('map50')} W={m.get('power_w_avg')}")
    return m


def system_info():
    info = {"platform": "Raspberry Pi 5 + AI HAT (Hailo-8L 13 TOPS)"}
    model = _vcgencmd(["version"])
    if model:
        info["firmware"] = model.splitlines()[0]
    try:
        import hailo_platform
        info["hailort"] = getattr(hailo_platform, "__version__", "?")
    except Exception:
        info["hailort"] = "yok"
    return info


def main():
    print("=" * 70)
    print("RASPBERRY PI 5 + HAILO-8L — benchmark")
    print("=" * 70)
    results = {"device_info": system_info(), "warmup": WARMUP_RUNS,
               "measure": MEASURE_RUNS, "models": []}
    for entry in REGISTRY:
        r = bench_one(entry)
        if r:
            results["models"].append(r)
    save_results(results, str(PROJECT_ROOT / "bechmarks"), "rpi5_hailo8l")


if __name__ == "__main__":
    main()

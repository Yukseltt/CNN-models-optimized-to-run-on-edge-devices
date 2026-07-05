# backends.py
#
# Platform algilama + her platform icin inference backend'i.
# Tum backend'ler ayni arayuzu sunar:
#     infer(image_bgr) -> (boxes_xyxy_pixel, scores, class_idx_0based)
#     class_names      : modelin cikti sirasindaki sinif isimleri (mAP eslemesi icin)
#     close()          : kaynaklari birak
#
# Backend tipleri:
#   ServerTorchBackend : CUDA/CPU torch baseline (yolo/frcnn/rtdetr; fp32/fp16)
#   TrtYoloBackend     : Jetson ultralytics .engine (TensorRT)
#   TrtOnnxBackend     : Jetson onnxruntime TensorRT EP (frcnn/rtdetr .onnx)
#   HailoBackend       : RPi5 + Hailo-8L .hef (HailoRT, int8)
#   CpuOnnxBackend     : RPi5 CPU onnxruntime baseline (Hailo'ya derlenemeyenler)

import os
import sys
from pathlib import Path

import numpy as np

from bench_core import letterbox
from config import CLASS_NAMES, PROJECT_ROOT, RTDETR_HF_CHECKPOINT

FRCNN_DIR = PROJECT_ROOT / "models" / "faster_rcnn"


# =====================================================================
# Platform algilama
# =====================================================================

def detect_platform() -> str:
    # "jetson" | "rpi5_hailo" | "server"
    # Cevre degiskeniyle ezilebilir: EDGE_PLATFORM=server|jetson|rpi5_hailo
    forced = os.environ.get("EDGE_PLATFORM")
    if forced in ("server", "jetson", "rpi5_hailo"):
        return forced

    model_str = ""
    try:
        model_str = Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except Exception:
        pass

    if "jetson" in model_str or "orin" in model_str or "nvidia" in model_str:
        return "jetson"
    if "raspberry" in model_str:
        return "rpi5_hailo"
    # Hailo cihazi takiliysa (ama Pi degilse de) hailo say.
    try:
        import hailo_platform  # noqa: F401
        return "rpi5_hailo"
    except Exception:
        pass
    return "server"


def count_torch_params(model) -> float:
    try:
        return sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        return None


def _empty():
    return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)


# =====================================================================
# SUNUCU (CUDA/CPU) torch baseline backend
# =====================================================================

class ServerTorchBackend:
    # type: "yolo" | "frcnn" | "rtdetr".  precision: "fp32" | "fp16".
    # Tek dosyadan (pt) yukler; pruned ckpt'ler 'model_obj' tutar (rebuild edilemez),
    # normal ckpt'ler state_dict + rebuild yolu izler.
    def __init__(self, mtype, pt_path, imgsz, precision="fp32", conf=0.3, device=None):
        import torch
        self.torch = torch
        self.mtype = mtype
        self.imgsz = imgsz
        self.conf = conf
        self.precision = precision
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = (precision == "fp16") and self.device.startswith("cuda")
        self.params_m = None
        self.class_names = list(CLASS_NAMES)

        if mtype == "yolo":
            self._init_yolo(pt_path)
        elif mtype == "frcnn":
            self._init_frcnn(pt_path)
        elif mtype == "rtdetr":
            self._init_rtdetr(pt_path)
        else:
            raise ValueError(f"bilinmeyen tip: {mtype}")

    # ---- YOLO (ultralytics) ----
    def _init_yolo(self, pt_path):
        from ultralytics import YOLO
        self.model = YOLO(str(pt_path))
        # YOLO model.names bu projede gt ile tutarsiz -> pozisyon-tabanli mAP
        # icin class_names'i CLASS_NAMES'te birakir (model_class_names verilmez).
        try:
            self.params_m = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        except Exception:
            pass

    def _infer_yolo(self, image_bgr):
        r = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf,
                               half=self.half, device=self.device,
                               verbose=False, save=False)[0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            return _empty()
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                b.cls.cpu().numpy().astype(int))

    # ---- Faster R-CNN (torchvision) ----
    def _init_frcnn(self, pt_path):
        torch = self.torch
        if str(FRCNN_DIR) not in sys.path:
            sys.path.insert(0, str(FRCNN_DIR))
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("model_obj") is not None:
            model = ckpt["model_obj"]                 # pruned: hazir nn.Module
            self.class_names = ckpt.get("class_names", list(CLASS_NAMES))
        else:
            from model_loaders import load_faster_rcnn  # normal: rebuild + state_dict
            model, class_names, _cfg = load_faster_rcnn(pt_path, "cpu")
            self.class_names = class_names
        # frcnn torchvision fp16'i guvenilir degil -> sunucuda hep fp32.
        self.half = False
        self.model = model.eval().to(self.device)
        self.params_m = count_torch_params(self.model)

    def _infer_frcnn(self, image_bgr):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).div_(255.0).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]   # torchvision: kutular ORIJINAL koordinatta
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        keep = scores >= self.conf
        # torchvision detection: 0=arka plan -> sinif 1-based; 0-based'e cek.
        return boxes[keep], scores[keep], (labels[keep] - 1).astype(int)

    # ---- RT-DETR v2 (HF) ----
    def _init_rtdetr(self, pt_path):
        torch = self.torch
        from transformers import AutoImageProcessor
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("model_obj") is not None:
            model = ckpt["model_obj"]                 # pruned: hazir nn.Module
            self.class_names = ckpt.get("class_names", list(CLASS_NAMES))
        else:
            from transformers import AutoConfig, AutoModelForObjectDetection
            class_names = ckpt.get("class_names", list(CLASS_NAMES))
            cfg = AutoConfig.from_pretrained(RTDETR_HF_CHECKPOINT)
            cfg.id2label = {i: n for i, n in enumerate(class_names)}
            cfg.label2id = {n: i for i, n in enumerate(class_names)}
            cfg.num_labels = len(class_names)
            model = AutoModelForObjectDetection.from_config(cfg)
            sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
            model.load_state_dict(sd, strict=False)
            self.class_names = class_names
        # KRITIK: egitim size'i (480) ile birebir; default 640 -> guven ~0.
        self.image_processor = AutoImageProcessor.from_pretrained(
            RTDETR_HF_CHECKPOINT, do_resize=True,
            size={"width": self.imgsz, "height": self.imgsz})
        model = model.eval().to(self.device)
        if self.half:
            model = model.half()
        self.model = model
        self.params_m = count_torch_params(self.model)

    def _infer_rtdetr(self, image_bgr):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        oh, ow = image_bgr.shape[:2]
        enc = self.image_processor(images=rgb, return_tensors="pt")
        pix = enc["pixel_values"].to(self.device)
        if self.half:
            pix = pix.half()
        with torch.no_grad():
            outputs = self.model(pixel_values=pix)
        target_sizes = torch.tensor([[oh, ow]], device=self.device)
        res = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.conf)[0]
        boxes = res["boxes"].float().cpu().numpy()
        scores = res["scores"].float().cpu().numpy()
        labels = res["labels"].cpu().numpy().astype(int)   # HF: 0-based
        return boxes, scores, labels

    # ---- dispatch ----
    def infer(self, image_bgr):
        if self.mtype == "yolo":
            return self._infer_yolo(image_bgr)
        if self.mtype == "frcnn":
            return self._infer_frcnn(image_bgr)
        return self._infer_rtdetr(image_bgr)

    def close(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# =====================================================================
# JETSON — ultralytics TensorRT .engine (YOLO)
# =====================================================================

class TrtYoloBackend:
    def __init__(self, artifact, imgsz, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(str(artifact))
        self.imgsz = imgsz
        self.conf = conf
        self.params_m = None
        self.class_names = list(CLASS_NAMES)   # pozisyon-tabanli mAP

    def infer(self, image_bgr):
        r = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf,
                               verbose=False, save=False)[0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            return _empty()
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                b.cls.cpu().numpy().astype(int))

    def close(self):
        pass


# =====================================================================
# JETSON — onnxruntime TensorRT EP (frcnn / rtdetr .onnx)
# =====================================================================

class TrtOnnxBackend:
    def __init__(self, artifact, imgsz, decode, precision="fp16", conf=0.3):
        import onnxruntime as ort
        cache = str(Path(artifact).parent / f"trt_cache_{Path(artifact).stem}")
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": precision in ("fp16", "int8"),
                "trt_int8_enable": precision == "int8",
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
            }),
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]
        self.sess = ort.InferenceSession(str(artifact), providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.decode = decode
        self.conf = conf
        self.params_m = None
        self.class_names = list(CLASS_NAMES)
        print(f"[trt-onnx] {Path(artifact).name} providers={self.sess.get_providers()}")

    def _pre(self, image_bgr):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.decode == "frcnn":
            img = cv2.resize(rgb, (self.imgsz, self.imgsz))
            t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            return t[None], 1.0, (0, 0), image_bgr.shape[:2]
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
        boxes = labels = scores = None
        for o in outs:
            if o.ndim == 2 and o.shape[1] == 4:
                boxes = o
            elif o.ndim == 1 and np.issubdtype(o.dtype, np.integer):
                labels = o
            elif o.ndim == 1:
                scores = o
        if boxes is None:
            return _empty()
        sx, sy = ow / self.imgsz, oh / self.imgsz
        boxes = boxes.copy(); boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
        keep = scores >= self.conf
        return boxes[keep], scores[keep], (labels[keep] - 1).astype(int)

    def _decode_rtdetr(self, outs, oh, ow):
        logits = boxes = None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
            elif o.ndim == 3:
                logits = o[0]
        if logits is None or boxes is None:
            return _empty()
        prob = 1.0 / (1.0 + np.exp(-logits))
        cls = prob.argmax(1); sc = prob.max(1)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([(cx - w / 2) * ow, (cy - h / 2) * oh,
                         (cx + w / 2) * ow, (cy + h / 2) * oh], 1)
        keep = sc >= self.conf
        return xyxy[keep], sc[keep], cls[keep].astype(int)

    def close(self):
        pass


# =====================================================================
# RPi5 + Hailo-8L — .hef (HailoRT, int8)
# =====================================================================

class HailoBackend:
    def __init__(self, artifact, imgsz, hailo_nms=True, conf=0.3):
        from hailo_platform import (VDevice, HEF, ConfigureParams,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams,
                                     FormatType)
        self.imgsz = imgsz
        self.hailo_nms = hailo_nms
        self.conf = conf
        self.params_m = None
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
        self.in_h, self.in_w = self.in_info.shape[0], self.in_info.shape[1]
        self._power_ok = False
        try:
            self.target.set_power_measurement()
            self.target.start_power_measurement()
            self._power_ok = True
        except Exception:
            pass
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
            return self._decode_nms(out, r, pw, ph)
        return _empty()

    def _decode_nms(self, out, r, pw, ph):
        dets = out[0] if isinstance(out, (list, tuple, np.ndarray)) and len(out) else out
        boxes, scores, labels = [], [], []
        H, W = self.in_h, self.in_w
        for cls_idx, arr in enumerate(dets):
            if arr is None or len(arr) == 0:
                continue
            for d in np.asarray(arr):
                ymin, xmin, ymax, xmax, sc = d[:5]
                if sc < self.conf:
                    continue
                x1 = (xmin * W - pw) / r; y1 = (ymin * H - ph) / r
                x2 = (xmax * W - pw) / r; y2 = (ymax * H - ph) / r
                boxes.append([x1, y1, x2, y2]); scores.append(float(sc)); labels.append(cls_idx)
        if not boxes:
            return _empty()
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
# RPi5 CPU — onnxruntime baseline (Hailo'ya derlenemeyen frcnn/rtdetr)
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
        self.params_m = None
        self.class_names = list(CLASS_NAMES)

    _pre = TrtOnnxBackend._pre
    _decode_frcnn = TrtOnnxBackend._decode_frcnn
    _decode_rtdetr = TrtOnnxBackend._decode_rtdetr

    def infer(self, image_bgr):
        blob, r, (pw, ph), (oh, ow) = self._pre(image_bgr)
        outs = self.sess.run(None, {self.inp: blob})
        if self.decode == "frcnn":
            return self._decode_frcnn(outs, oh, ow)
        return self._decode_rtdetr(outs, oh, ow)

    def close(self):
        pass

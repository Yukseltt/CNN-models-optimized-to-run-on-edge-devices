# speed_runner.py
#
# Inference latency measurement with production settings.
# Production ayarlari ile inference gecikme olcumu.
#
# Protocol / Protokol:
#     - Batch size = 1 (edge device scenario)
#     - Batch size = 1 (edge cihaz senaryosu)
#     - 20 warmup inferences (not measured)
#     - 20 warmup inference (olculmuyor)
#     - 100 measured inferences with torch.cuda.synchronize
#     - 100 olculen inference, torch.cuda.synchronize ile
#     - YOLO: conf=0.25 (production default)
#     - Faster R-CNN: default score threshold

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

WARMUP_RUNS   = 20
MEASURE_RUNS  = 100


def _load_test_image_path(test_images_dir: str) -> str:
    # Get the first image from test set for latency measurement.
    # Gecikme olcumu icin test setinden ilk goruntuyu al.
    img_dir = Path(test_images_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in exts:
            return str(f)
    raise FileNotFoundError(f"No image found in {test_images_dir}")


def _prepare_frcnn_input(image_path: str, device: str) -> torch.Tensor:
    # Read and preprocess an image for Faster R-CNN inference.
    # Faster R-CNN inference icin goruntuyu oku ve isle.
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def run_yolo_speed(model, test_images_dir: str) -> dict:
    # Measure YOLO inference latency at production conf=0.25.
    # Production conf=0.25'te YOLO inference gecikmesini olc.
    img_path = _load_test_image_path(test_images_dir)

    # Warmup.
    # Isinma.
    for _ in range(WARMUP_RUNS):
        model.predict(source=img_path, conf=0.25, save=False, verbose=False)

    # Measure.
    # Olcum.
    torch.cuda.synchronize()
    times = []
    for _ in range(MEASURE_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=img_path, conf=0.25, save=False, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return _compute_stats(times)


def run_frcnn_speed(model: nn.Module, test_images_dir: str, device: str = "cuda:0") -> dict:
    # Measure Faster R-CNN inference latency.
    # Faster R-CNN inference gecikmesini olc.
    img_path = _load_test_image_path(test_images_dir)
    img_tensor = _prepare_frcnn_input(img_path, device)

    model.eval()

    # Warmup.
    # Isinma.
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model([img_tensor])

    # Measure.
    # Olcum.
    torch.cuda.synchronize()
    times = []
    for _ in range(MEASURE_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model([img_tensor])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return _compute_stats(times)


def _compute_stats(times: list) -> dict:
    # Compute latency statistics from a list of measurements.
    # Olcum listesinden gecikme istatistiklerini hesaplar.
    arr = np.array(times)
    mean = float(arr.mean())
    return {
        "latency_mean_ms": round(mean, 2),
        "latency_std_ms":  round(float(arr.std()), 2),
        "latency_min_ms":  round(float(arr.min()), 2),
        "latency_max_ms":  round(float(arr.max()), 2),
        "fps":             round(1000.0 / mean, 1) if mean > 0 else 0.0,
    }
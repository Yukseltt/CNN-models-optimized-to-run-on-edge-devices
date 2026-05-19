# determinism_checker.py
#
# Verify that model inference is deterministic.
# Model inference'inin deterministik oldugunu dogrula.
#
# Runs the same image twice and checks if predictions are identical.
# Ayni goruntuyu iki kez calistirir ve tahminlerin ayni olup olmadigini kontrol eder.

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


def _load_test_image_path(test_images_dir: str) -> str:
    img_dir = Path(test_images_dir)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in exts:
            return str(f)
    raise FileNotFoundError(f"No image found in {test_images_dir}")


def _check_yolo_determinism(model, test_images_dir: str) -> dict:
    # Check YOLO inference determinism.
    # YOLO inference determinizmini kontrol et.
    img_path = _load_test_image_path(test_images_dir)

    r1 = model.predict(source=img_path, conf=0.25, save=False, verbose=False)
    r2 = model.predict(source=img_path, conf=0.25, save=False, verbose=False)

    if len(r1) == 0 or len(r2) == 0:
        return {"deterministic": len(r1) == len(r2), "max_diff": 0.0}

    b1 = r1[0].boxes
    b2 = r2[0].boxes

    if b1 is None and b2 is None:
        return {"deterministic": True, "max_diff": 0.0}

    if b1 is None or b2 is None:
        return {"deterministic": False, "max_diff": float("inf")}

    if len(b1) != len(b2):
        return {"deterministic": False, "max_diff": float("inf")}

    xyxy1 = b1.xyxy.cpu().numpy()
    xyxy2 = b2.xyxy.cpu().numpy()
    max_diff = float(np.abs(xyxy1 - xyxy2).max())

    return {
        "deterministic": max_diff < 1e-5,
        "max_diff":      round(max_diff, 8),
    }


def _check_frcnn_determinism(
    model: nn.Module, test_images_dir: str, device: str = "cuda:0",
) -> dict:
    # Check Faster R-CNN inference determinism.
    # Faster R-CNN inference determinizmini kontrol et.
    img_path = _load_test_image_path(test_images_dir)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        p1 = model([tensor])
        p2 = model([tensor])

    b1 = p1[0]["boxes"].cpu().numpy()
    b2 = p2[0]["boxes"].cpu().numpy()

    if len(b1) != len(b2):
        return {"deterministic": False, "max_diff": float("inf")}

    if len(b1) == 0:
        return {"deterministic": True, "max_diff": 0.0}

    max_diff = float(np.abs(b1 - b2).max())

    return {
        "deterministic": max_diff < 1e-5,
        "max_diff":      round(max_diff, 8),
    }


def check_determinism(
    model, family: str, test_images_dir: str, device: str = "cuda:0",
) -> dict:
    # Unified determinism check for both model families.
    # Her iki model ailesi icin birlesik determinizm kontrolu.
    if family == "yolo":
        return _check_yolo_determinism(model, test_images_dir)
    else:
        return _check_frcnn_determinism(model, test_images_dir, device)
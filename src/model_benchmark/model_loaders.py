# model_loaders.py
#
# Unified model loader for YOLO (v8, 11, 26) and Faster R-CNN families.
# YOLO (v8, 11, 26) ve Faster R-CNN aileleri icin birlesik model yukleyici.
#
# Auto-detects model family from run directory name.
# Model ailesini run dizin adindan otomatik algilar.

import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRCNN_DIR = PROJECT_ROOT / "models" / "faster_rcnn"


def detect_family(run_dir: Path) -> str:
    # Detect model family from directory name.
    # Dizin adindan model ailesini algilar.
    name = run_dir.name.lower()
    if name.startswith("faster_rcnn"):
        return "frcnn"
    return "yolo"


def find_best_pt(run_dir: Path, family: str) -> Optional[Path]:
    # Locate best.pt inside a run directory.
    # Run dizininde best.pt dosyasini bulur.
    if family == "frcnn":
        pt = run_dir / "best.pt"
    else:
        pt = run_dir / "weights" / "best.pt"
    return pt if pt.exists() else None


def load_yolo(best_pt: Path) -> object:
    # Load a YOLO model (v8, 11, or 26) via Ultralytics API.
    # Ultralytics API ile YOLO modelini (v8, 11, 26) yukler.
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    return model


def load_faster_rcnn(best_pt: Path, device: str = "cuda:0") -> nn.Module:
    # Load a Faster R-CNN model from our custom checkpoint.
    # Bizim custom checkpoint'imizden Faster R-CNN modelini yukler.
    if str(FRCNN_DIR) not in sys.path:
        sys.path.insert(0, str(FRCNN_DIR))
    from model_factory import create_faster_rcnn

    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    backbone = cfg.get("BACKBONE", "resnet50")
    pretrained = cfg.get("PRETRAINED", "coco")
    class_names = ckpt.get("class_names", ["Person", "Car", "OtherVehicle"])
    num_classes = len(class_names) + 1

    model = create_faster_rcnn(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    return model, class_names, cfg


def load_model(run_dir: Path, device: str = "cuda:0") -> dict:
    # Unified loader. Returns a dict with model and metadata.
    # Birlesik yukleyici. Model ve metadata iceren dict doner.
    family = detect_family(run_dir)
    best_pt = find_best_pt(run_dir, family)

    if best_pt is None:
        return None

    info = {
        "run_dir":   run_dir,
        "run_name":  run_dir.name,
        "family":    family,
        "best_pt":   best_pt,
        "device":    device,
    }

    if family == "yolo":
        model = load_yolo(best_pt)
        info["model"] = model
        info["class_names"] = list(model.names.values()) if hasattr(model, "names") else []
    else:
        model, class_names, cfg = load_faster_rcnn(best_pt, device)
        info["model"] = model
        info["class_names"] = class_names
        info["frcnn_cfg"] = cfg

    return info


def discover_runs(runs_dir: Path, skip_patterns: Optional[list] = None) -> list[Path]:
    # Find all valid run directories under runs_dir.
    # runs_dir altindaki tum gecerli run dizinlerini bulur.
    if skip_patterns is None:
        skip_patterns = [
            "detect", "efficientnet", "unet",
            "sayzek_runs_yolo11_base_param_puring",
            "thermal_yolo",
        ]

    result = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        skip = False
        for pat in skip_patterns:
            # Exact match for patterns that could be substrings of valid names.
            # Gecerli isimlerin alt dizisi olabilecek pattern'lar icin tam esleme.
            if pat == d.name:
                skip = True
                break
            # Substring match for broad category patterns (e.g. "detect", "efficientnet").
            # Genis kategori pattern'lari icin alt dizi eslesmesi (orn. "detect", "efficientnet").
            if "/" not in pat and len(pat) < 30 and pat in d.name:
                skip = True
                break
        if skip:
            continue
        family = detect_family(d)
        pt = find_best_pt(d, family)
        if pt is not None:
            result.append(d)

    return result
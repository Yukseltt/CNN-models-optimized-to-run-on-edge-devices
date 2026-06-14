# model_loaders.py
#
# Unified model loader for every detection family that lands in runs/.
# runs/ altina dusen her detection ailesi icin birlesik model yukleyici.
#
# Families / Aileler:
#     - "yolo"   : Ultralytics (v8 / 11 / 26, YOLO-FIR, YOLO-Thermal).
#     - "frcnn"  : Torchvision Faster R-CNN (custom checkpoint).
#     - "ssd"    : Torchvision SSDLite320 + MobileNetV3.
#     - "fcos"   : Torchvision FCOS ResNet50-FPN (anchor-free).
#     - "mlpd"   : Torchvision RetinaNet ResNet50-FPN-v2 (thermal MLPD).
#     - "rtdetr" : HuggingFace RT-DETR v2 (wrapped to Torchvision contract).
#     - "detr"   : HuggingFace DETR (wrapped to Torchvision contract).
#
# Everything except "yolo" exposes the Torchvision detection contract
# (model(list_of_CHW_float01_tensors) -> [{"boxes","labels","scores"}]),
# so the Faster R-CNN accuracy / speed / memory / determinism runners are
# reused unchanged for the whole non-yolo group.
# "yolo" disindaki her sey Torchvision detection kontratini sunar
# (model(CHW_float01_tensor_listesi) -> [{"boxes","labels","scores"}]),
# boylece Faster R-CNN dogruluk / hiz / bellek / determinizm runner'lari
# tum non-yolo grup icin degismeden yeniden kullanilir.

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT  = PROJECT_ROOT / "models"
FRCNN_DIR    = MODELS_ROOT / "faster_rcnn"

# Torchvision detection families: directory-name prefix -> (models/<dir>, factory fn).
# Torchvision detection aileleri: dizin-adi prefix -> (models/<dir>, factory fn).
_TV_FAMILIES = {
    "frcnn": ("faster_rcnn", "create_faster_rcnn"),
    "ssd":   ("ssd",         "create_ssd"),
    "fcos":  ("fcos",        "create_fcos"),
    "mlpd":  ("mlpd",        "create_mlpd"),
}

# HuggingFace detection families (wrapped to Torchvision contract).
# HuggingFace detection aileleri (Torchvision kontratina sarilir).
_HF_FAMILIES = {"rtdetr", "detr"}


def detect_family(run_dir: Path) -> str:
    # Detect model family from the run directory name.
    # Run dizin adindan model ailesini algilar.
    #
    # Order matters: "rtdetr" must be tested before "detr" (substring), and
    # "ssdlite"/"ssd" before the generic fallback.
    # Sira onemli: "rtdetr", "detr"den once test edilmeli (alt dizi), ve
    # "ssdlite"/"ssd" genel fallback'ten once.
    name = run_dir.name.lower()
    if name.startswith("faster_rcnn"):
        return "frcnn"
    if name.startswith(("ssdlite", "ssd")):
        return "ssd"
    if name.startswith("fcos"):
        return "fcos"
    if name.startswith("mlpd"):
        return "mlpd"
    if name.startswith("rtdetr"):
        return "rtdetr"
    if name.startswith("detr"):
        return "detr"
    return "yolo"


def find_best_pt(run_dir: Path, family: str) -> Optional[Path]:
    # Locate best.pt inside a run directory.
    # Run dizininde best.pt dosyasini bulur.
    #
    # YOLO keeps weights under weights/, everyone else at the run root.
    # YOLO agirliklari weights/ altinda tutar, digerleri run kokunde.
    if family == "yolo":
        pt = run_dir / "weights" / "best.pt"
    else:
        pt = run_dir / "best.pt"
    return pt if pt.exists() else None


def _load_factory(model_subdir: str, func_name: str):
    # Load create_* from models/<subdir>/model_factory.py under a unique
    # module name so the several same-named model_factory.py files never
    # shadow each other in sys.modules.
    # models/<subdir>/model_factory.py icinden create_*'i benzersiz bir modul
    # adiyla yukler; boylece ayni adli birden cok model_factory.py dosyasi
    # sys.modules'ta birbirini golgelemez.
    path = MODELS_ROOT / model_subdir / "model_factory.py"
    spec = importlib.util.spec_from_file_location(
        f"_benchmark_factory_{model_subdir}", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


def load_yolo(best_pt: Path) -> object:
    # Load a YOLO model (v8, 11, 26, FIR, Thermal) via the Ultralytics API.
    # Ultralytics API ile bir YOLO modeli (v8, 11, 26, FIR, Thermal) yukler.
    from ultralytics import YOLO
    return YOLO(str(best_pt))


def load_torchvision(best_pt: Path, family: str, device: str = "cuda:0"):
    # Load any Torchvision detection family from our custom checkpoint.
    # Bizim custom checkpoint'imizden herhangi bir Torchvision detection
    # ailesini yukler.
    #
    # Checkpoint schema (shared across frcnn / ssd / fcos / mlpd):
    #     {epoch, model(state_dict), cfg, class_names, metric}
    # Checkpoint semasi (frcnn / ssd / fcos / mlpd ortak):
    #     {epoch, model(state_dict), cfg, class_names, metric}
    model_subdir, func_name = _TV_FAMILIES[family]
    create_fn = _load_factory(model_subdir, func_name)

    ckpt = torch.load(best_pt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    class_names = ckpt.get("class_names", ["Person", "Car", "OtherVehicle"])
    # Every train script builds the model with num_classes = len(names) + 1
    # (background slot, even for the sigmoid-head detectors).
    # Her train script modeli num_classes = len(names) + 1 ile kurar
    # (background slot, sigmoid-head detector'lar icin bile).
    num_classes = len(class_names) + 1

    backbone   = cfg.get("BACKBONE") or cfg.get("STUDENT_BACKBONE") or "resnet50"
    pretrained = cfg.get("PRETRAINED", "coco")

    model = create_fn(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    return model, class_names, cfg


def load_hf_detector(best_pt: Path, family: str, device: str = "cuda:0"):
    # Load a HuggingFace detector (RT-DETR / DETR) and wrap it to the
    # Torchvision contract via HFDetectorWrapper.
    # Bir HuggingFace detector (RT-DETR / DETR) yukler ve HFDetectorWrapper
    # ile Torchvision kontratina sarar.
    #
    # Checkpoint schema mirrors the Torchvision one:
    #     {epoch, model(state_dict), cfg, class_names, metric}
    # The architecture/config + preprocessing come from cfg["CHECKPOINT"]
    # (the HF Hub id), with only the label maps overridden to our classes.
    # Checkpoint semasi Torchvision ile ayni:
    #     {epoch, model(state_dict), cfg, class_names, metric}
    # Mimari/config + preprocessing cfg["CHECKPOINT"]'ten (HF Hub id) gelir,
    # sadece label haritalari kendi siniflarimiza gore override edilir.
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForObjectDetection,
    )
    from src.model_benchmark.hf_detector import HFDetectorWrapper

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    class_names = ckpt.get("class_names", ["Person", "Car", "OtherVehicle"])

    hub_id     = cfg.get("CHECKPOINT")
    image_size = cfg.get("IMAGE_SIZE", 640)

    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in enumerate(class_names)}

    config = AutoConfig.from_pretrained(hub_id)
    config.id2label   = id2label
    config.label2id   = label2id
    config.num_labels = len(class_names)

    model = AutoModelForObjectDetection.from_config(config)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    image_processor = AutoImageProcessor.from_pretrained(
        hub_id,
        do_resize=True,
        size={"width": image_size, "height": image_size},
        use_fast=True,
    )

    wrapper = HFDetectorWrapper(model, image_processor).to(device)
    return wrapper, class_names, cfg


def load_model(run_dir: Path, device: str = "cuda:0") -> Optional[dict]:
    # Unified loader. Returns a dict with model and metadata, or None if no
    # finished checkpoint (best.pt) is present.
    # Birlesik yukleyici. Model ve metadata iceren dict doner, bitmis bir
    # checkpoint (best.pt) yoksa None.
    family  = detect_family(run_dir)
    best_pt = find_best_pt(run_dir, family)
    if best_pt is None:
        return None

    info = {
        "run_dir":  run_dir,
        "run_name": run_dir.name,
        "family":   family,
        "best_pt":  best_pt,
        "device":   device,
    }

    if family == "yolo":
        model = load_yolo(best_pt)
        info["model"] = model
        info["class_names"] = list(model.names.values()) if hasattr(model, "names") else []
    elif family in _TV_FAMILIES:
        model, class_names, cfg = load_torchvision(best_pt, family, device)
        info["model"] = model
        info["class_names"] = class_names
        info["cfg"] = cfg
    elif family in _HF_FAMILIES:
        model, class_names, cfg = load_hf_detector(best_pt, family, device)
        info["model"] = model
        info["class_names"] = class_names
        info["cfg"] = cfg
    else:
        return None

    return info


def discover_runs(runs_dir: Path, skip_patterns: Optional[list] = None) -> list:
    # Find every run directory under runs_dir that has a finished checkpoint.
    # runs_dir altinda bitmis checkpoint'i olan her run dizinini bulur.
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
            # Substring match for broad category patterns (e.g. "detect").
            # Genis kategori pattern'lari icin alt dizi eslesmesi (orn. "detect").
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

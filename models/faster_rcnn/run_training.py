# run_training.py
#
# Launcher script that starts Faster R-CNN training.
# Faster R-CNN egitimini baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# Switch between backbones by changing the BACKBONE_TYPE variable below.
# Backbone'lar arasi gecis icin asagidaki BACKBONE_TYPE degiskenini degistir.

import sys
from pathlib import Path

# Add project root to sys.path for any cross-module imports.
# Cross-module import'lar icin proje kokunu sys.path'e ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add current directory for local module imports.
# Lokal modul import'lari icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))

from train_faster_rcnn import train


# Backbone selector. Set to "resnet50" or "mobilenet" before running.
# Backbone secici. Calistirmadan once "resnet50" veya "mobilenet" yap.
BACKBONE_TYPE = "mobilenet"

# Shared / Ortak

DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_coco_dataset/dataset_augmented"


# ResNet50-FPN configuration / ResNet50-FPN konfigurasyonu

RESNET50_CFG = {
    "BACKBONE":    "resnet50",
    "BATCH_SIZE":  8,
    "LR0":         0.005,
    "NAME":        "faster_rcnn_resnet50_28.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}



# MobileNetV3-FPN configuration / MobileNetV3-FPN konfigurasyonu

MOBILENET_CFG = {
    "BACKBONE":    "mobilenet",
    "BATCH_SIZE":  16,
    "LR0":         0.01,
    "NAME":        "faster_rcnn_mobilenet_28.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
    "RESUME_FROM": "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs/faster_rcnn_mobilenet_29.04.2026/last.pt",
}


# Dispatch / Yonlendirme

if __name__ == "__main__":
    if BACKBONE_TYPE == "resnet50":
        train(data_dir=DATA_DIR, cfg=RESNET50_CFG)
    elif BACKBONE_TYPE == "mobilenet":
        train(data_dir=DATA_DIR, cfg=MOBILENET_CFG)
    else:
        raise ValueError(
            f"BACKBONE_TYPE must be 'resnet50' or 'mobilenet', got: {BACKBONE_TYPE!r}"
        )
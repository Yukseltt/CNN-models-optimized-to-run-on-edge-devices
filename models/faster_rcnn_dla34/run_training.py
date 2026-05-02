# run_training.py
#
# Launcher for Faster R-CNN with DLA-34 + FPN backbone.
# DLA-34 + FPN backbone'lu Faster R-CNN icin baslatici.
#
# Usage / Kullanim:
#     python run_training.py

import sys
from pathlib import Path

# Add project root to sys.path for any cross-module imports.
# Cross-module import'lar icin proje kokunu sys.path'e ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add current directory for local module imports.
# Lokal modul import'lari icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))

from train_faster_rcnn_dla34 import train


# Data directory / Veri dizini.

DATA_DIR = str(PROJECT_ROOT / "dataset" / "dataset_augmented_04_23_2026" / "dataset_augmented")


# DLA-34 + FPN configuration / DLA-34 + FPN konfigurasyonu.

DLA34_CFG = {
    "EPOCHS":        2,
    "BATCH_SIZE":    2,
    "LR0":           0.005,
    "WARMUP_EPOCHS": 1,
    "NAME":          "faster_rcnn_dla34_02.05.2026",
    "OUTPUT_XLSX":   "training_metrics.xlsx",
}


if __name__ == "__main__":
    train(data_dir=DATA_DIR, cfg=DLA34_CFG)

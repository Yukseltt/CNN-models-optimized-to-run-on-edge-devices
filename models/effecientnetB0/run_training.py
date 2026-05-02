# run_training.py
#
# Launcher for EfficientNet-B0 thermal bbox classifier.
# EfficientNet-B0 termal bbox siniflandirici icin baslatici.
#
# Usage / Kullanim:
#     python run_training.py

import sys
from pathlib import Path

# Proje kokunu sys.path'e ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lokal modul import'lari icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))

from train_efficientnet_b0 import train


# Veri dizini.
DATA_DIR = str(PROJECT_ROOT / "dataset" / "dataset_augmented_04_23_2026" / "dataset_augmented")


# EfficientNet-B0 konfigurasyonu.
EFFB0_CFG = {
    "EPOCHS":         50,
    "BATCH_SIZE":     64,
    "LR0":            0.01,
    "WARMUP_EPOCHS":  2,
    "PATIENCE":       10,
    "USE_AMP":        True,
    "NAME":           "efficientnet_b0_02.05.2026",
    "OUTPUT_XLSX":    "training_metrics.xlsx",
}


if __name__ == "__main__":
    train(data_dir=DATA_DIR, cfg=EFFB0_CFG)

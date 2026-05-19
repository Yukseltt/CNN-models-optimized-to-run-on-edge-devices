# run_training.py
#
# Launcher for U-Net thermal segmentation training.
# U-Net termal segmentasyon egitimi icin baslatici.
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

from train_unet import train


# Veri dizini.
DATA_DIR = str(PROJECT_ROOT / "dataset" / "2x_augmented_unet_dataset")


# U-Net konfigurasyonu.
UNET_CFG = {
    "EPOCHS":            300,
    "BATCH_SIZE":        64,
    "LR0":               0.01,
    "WARMUP_EPOCHS":     2,
    "PATIENCE":          15,
    "USE_AMP":           True,

    "ENCODER_NAME":      "resnet34",
    "ENCODER_WEIGHTS":   "imagenet",
    "INPUT_SIZE":        512,
    "NUM_CLASSES":       4,           # bg + Person + Car + OtherVehicle

    "USE_CLASS_WEIGHTS": True,        # sinif dengesizligi remedy

    "NAME":              "unet_resnet34_03.05.2026",
    "OUTPUT_XLSX":       "training_metrics.xlsx",
}


if __name__ == "__main__":
    train(data_dir=DATA_DIR, cfg=UNET_CFG)

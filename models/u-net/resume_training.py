# resume_training.py
#
# Resume launcher for U-Net thermal segmentation training.
# U-Net termal segmentasyon egitimi icin devam ettirici.
#
# Yarida kesilen son egitimi last.pt'den devam ettirir; ayni run klasorune
# yazmaya devam eder, training_metrics.xlsx dosyasini gunceller.
#
# Usage / Kullanim:
#     python resume_training.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_unet import train


DATA_DIR = str(PROJECT_ROOT / "dataset" / "2x_augmented_unet_dataset")

# Devam edilecek run klasoru.
RESUME_RUN_DIR = PROJECT_ROOT / "runs" / "unet_resnet34_03.05.20264"
RESUME_CKPT    = RESUME_RUN_DIR / "last.pt"


# run_training.py ile ayni egitim hiperparametreleri.
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
    "NUM_CLASSES":       4,

    "USE_CLASS_WEIGHTS": True,

    "NAME":              "unet_resnet34_03.05.2026",
    "OUTPUT_XLSX":       "training_metrics.xlsx",

    "RESUME_FROM":       str(RESUME_CKPT),
}


if __name__ == "__main__":
    if not RESUME_CKPT.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")
    train(data_dir=DATA_DIR, cfg=UNET_CFG)

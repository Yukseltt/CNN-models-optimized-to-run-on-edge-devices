# resume_training.py
#
# Resume launcher for EfficientNet-B0 thermal bbox classifier.
# EfficientNet-B0 termal bbox siniflandirici icin devam ettirici.
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

from train_efficientnet_b0 import train


DATA_DIR = str(PROJECT_ROOT / "dataset" / "2x_augmented_coco_dataset" / "dataset_augmented")

RESUME_RUN_DIR = PROJECT_ROOT / "runs" / "efficientnet_b0_02.05.20266"
RESUME_CKPT    = RESUME_RUN_DIR / "last.pt"


# run_training.py ile ayni egitim hiperparametreleri.
EFFB0_CFG = {
    "EPOCHS":         300,
    "BATCH_SIZE":     128,
    "LR0":            0.02,
    "WARMUP_EPOCHS":  1,
    "PATIENCE":       10,
    "USE_AMP":        True,
    "NAME":           "efficientnet_b0_02.05.2026",
    "OUTPUT_XLSX":    "training_metrics.xlsx",

    "RESUME_FROM":    str(RESUME_CKPT),
}


if __name__ == "__main__":
    if not RESUME_CKPT.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")
    train(data_dir=DATA_DIR, cfg=EFFB0_CFG)

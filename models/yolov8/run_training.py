# run_training.py
#
# Launcher script that starts the training.
# Egitimi baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# All parameters are defined here.
# Tum parametreler burada tanimlanir.

import sys
from pathlib import Path

# Add project root to sys.path so src.cb_loss can be imported.
# src.cb_loss import edilebilmesi icin proje kokunu sys.path'e ekle.
# models/yolov8/run_training.py -> uc_cihazlarda_terhmal_object_detection/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add current directory for train_yolov8_nano_cb import.
# train_yolov8_nano_cb import icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))

from train_yolov8_nano_cb import train


# Dataset / Veri seti

DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml"

# Per-class training sample counts in the same order as data.yaml names.
# data.yaml names sirasi ile ayni sirada per-class egitim ornek sayilari.
CLASS_COUNTS = [335428, 577119, 36186]

CFG_OVERRIDE = {
    "BETA":        0.999999,
    "NAME":        "yolov8n_cb_beta_0.999999_augmented_24.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


if __name__ == "__main__":
    train(
        data=DATA_YAML,
        class_counts=CLASS_COUNTS,
        cfg=CFG_OVERRIDE,
    )
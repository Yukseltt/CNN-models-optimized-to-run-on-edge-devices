# run_teacher_training.py
#
# Launcher for YOLOv8-large teacher training (distillation teacher).
# YOLOv8-large teacher egitimi launcher'i (distillation teacher).
#
# Usage / Kullanim:
#     python run_teacher_training.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_yolov8_large_default import train as train_large


# Shared / Ortak
DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/restratified_yolo/data.yaml"


# Teacher configuration / Teacher konfigurasyonu
#
# All hyperparameters use Ultralytics defaults (same as nano).
# Tum hyperparametreler Ultralytics varsayilanlarini kullanir (nano ile ayni).
# Only batch size is lower (8) due to the larger model footprint.
# Sadece batch size daha dusuk (8), buyuk model ayak izi nedeniyle.

TEACHER_CFG = {
    "BATCH_SIZE":  8,
    "NAME":        "yolov8l_restratified_test_14.06.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


if __name__ == "__main__":
    train_large(data=DATA_YAML, cfg=TEACHER_CFG)
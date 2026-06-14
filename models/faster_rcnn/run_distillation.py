# run_distillation.py
#
# Launcher for Faster R-CNN knowledge distillation.
# Faster R-CNN bilgi damitma launcher'i.
#
# Usage / Kullanim:
#     python run_distillation.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_faster_rcnn import distill


# Shared / Ortak

DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_coco_dataset/dataset_augmented"

RUNS_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs"


# Teacher and student checkpoints (warm start).
# Teacher ve student checkpoint'leri (warm start).

TEACHER_CKPT = f"{RUNS_DIR}/faster_rcnn_resnet50_28.04.2026/best.pt"
STUDENT_CKPT = f"{RUNS_DIR}/faster_rcnn_mobilenet_29.04.2026/best.pt"


# Distillation configuration / Damitma konfigurasyonu.

DISTILL_CFG = {
    "TEACHER_BACKBONE":  "resnet50",
    "STUDENT_BACKBONE":  "mobilenet",
    "TEACHER_CKPT":      TEACHER_CKPT,
    "STUDENT_CKPT":      STUDENT_CKPT,

    "BATCH_SIZE":        4,
    "LR0":               0.005,
    "EPOCHS":            40,
    "PATIENCE":          15,

    "TEMPERATURE":       3.0,
    "BETA_CLS":          0.7,
    "BETA_BOX":          0.5,

    "NAME":              "faster_rcnn_distill_r50_to_mobilenet_12.06.2026",
    "OUTPUT_XLSX":       "training_metrics.xlsx",
}


if __name__ == "__main__":
    distill(data_dir=DATA_DIR, cfg=DISTILL_CFG)
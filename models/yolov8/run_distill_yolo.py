# run_distill_yolo.py
#
# Launcher for YOLOv8l -> YOLOv8n feature distillation.
# YOLOv8l -> YOLOv8n feature damitma launcher'i.
#
# Usage / Kullanim:
#     python run_distill_yolo.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_yolo import distill


TEACHER_PT = str(PROJECT_ROOT / "runs" / "yolov8l_restratified_test_14.06.2026" / "weights" / "best.pt")
STUDENT_PT = str(PROJECT_ROOT / "runs_new" / "yolov8n_default_restratified_14.06.2026" / "weights" / "best.pt")
DATA_YAML  = str(PROJECT_ROOT / "dataset" / "restratified_yolo" / "data.yaml")
RUNS_NEW   = str(PROJECT_ROOT / "runs_new")


DISTILL_CFG = {
    "TEACHER_PT": TEACHER_PT,
    "STUDENT_PT": STUDENT_PT,
    "DATA_YAML":  DATA_YAML,
    "PROJECT":    RUNS_NEW,
    "NAME":       "yolov8n_distill_from_l_19.06.2026",

    "BATCH_SIZE": 16,
    "EPOCHS":     50,
    "PATIENCE":   15,
    "LR0":        0.001,
    "ALPHA_FEAT": 1.0,
}


if __name__ == "__main__":
    distill(cfg=DISTILL_CFG)
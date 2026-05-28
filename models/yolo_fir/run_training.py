# run_training.py
#
# Launcher script that starts YOLO-FIR training.
# YOLO-FIR egitimini baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# Switch configurations by changing the RUN_TYPE variable below.
# Konfigurasyonlar arasi gecis icin asagidaki RUN_TYPE'i degistir.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).parent))

from train_yolo_fir import train


# Run type selector. One of:
# Run tipi seciciyi. Sunlardan biri:
#   "default"  - YOLOv5su base + FIR-paper-inspired config
#   "resume"   - resume an interrupted run from last.pt

RUN_TYPE = "default"


# Shared / Ortak
DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml"


# Default FIR config / Varsayilan FIR config.
DEFAULT_CFG = {
    "EPOCHS":      100,
    "BATCH_SIZE":  32,
    "PATIENCE":    30,
    "NAME":        "yolo_fir_v5s_28.05.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Resume config / Resume config.
RESUME_CFG = {
    "EPOCHS":       100,
    "BATCH_SIZE":   32,
    "PATIENCE":     30,
    "NAME":         "yolo_fir_v5s_28.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
    "RESUME_FROM":  str(PROJECT_ROOT / "runs" / "yolo_fir_v5s_28.05.2026" / "weights" / "last.pt"),
}


CONFIGS = {
    "default": DEFAULT_CFG,
    "resume":  RESUME_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )
    train(data=DATA_YAML, cfg=CONFIGS[RUN_TYPE])

# run_yolov8n_restratified.py
#
# Launcher for YOLOv8-nano training on the restratified dataset.
# Restratified dataset uzerinde YOLOv8-nano egitimi launcher'i.
#
# Outputs go under runs_new/ to keep them separate from old (invalid) runs.
# Ciktilar eski (gecersiz) run'lardan ayri tutmak icin runs_new/ altina gider.
#
# Usage / Kullanim:
#     python run_yolov8n_restratified.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_yolov8_nano_default import train as train_nano


# Restratified dataset (corrected split + corrected data.yaml).
# Restratified dataset (duzeltilmis bolme + duzeltilmis data.yaml).
DATA_YAML = str(PROJECT_ROOT / "dataset" / "restratified_yolo" / "data.yaml")

# New runs directory, separate from old invalid results.
# Yeni runs dizini, eski gecersiz sonuclardan ayri.
RUNS_NEW = str(PROJECT_ROOT / "runs_new")


# Configuration / Konfigurasyon
#
# Same defaults as the original nano training, only data path and output
# directory change.
# Orijinal nano egitimiyle ayni varsayilanlar, sadece veri yolu ve cikti
# dizini degisir.

NANO_CFG = {
    "PROJECT": RUNS_NEW,
    "NAME":    "yolov8n_default_restratified_14.06.2026",
}


if __name__ == "__main__":
    train_nano(data=DATA_YAML, cfg=NANO_CFG)
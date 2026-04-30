# run_training.py
#
# Launcher script that starts training with CB Loss, Focal Loss, or default mode.
# CB Loss, Focal Loss veya default mode ile egitimi baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# Switch between loss types by changing the LOSS_TYPE variable below.
# Loss tipini degistirmek icin asagidaki LOSS_TYPE degiskenini degistir.

import sys
from pathlib import Path

# Add project root to sys.path for src imports.
# src import'lari icin proje kokunu sys.path'e ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add current directory for trainer imports.
# Trainer import'lari icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))


# Loss type selector. Set to "cb", "focal" or "default" before running.
# Loss tipi seciciyi calistirmadan once "cb", "focal" veya "default" olarak ayarla.
LOSS_TYPE = "default"


# Shared / Ortak
DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml"


# CB Loss configuration / CB Loss konfigurasyonu

CLASS_COUNTS = [335428, 577119, 36186]

CB_CFG = {
    "BETA":        0.99999999,
    "NAME":        "yolov8n_cb_beta_0.99999999_augmented_29.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Focal Loss configuration / Focal Loss konfigurasyonu

FCL_CFG = {
    "FCL_ALPHA":   0.25,
    "FCL_GAMMA":   2.0,
    "NAME":        "yolov8n_fcl_alpha_0.25_gamma_2.0_augmented_29.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Default configuration / Default konfigurasyon
#
# All hyperparameters use Ultralytics defaults.
# Tum hyperparametreler Ultralytics varsayilanlarini kullanir.
# Only run name and output filename are overridden.
# Sadece run ismi ve cikti dosya adi override edilir.

DEFAULT_CFG = {
    "NAME":        "yolov8n_default_augmented_29.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Dispatch / Yonlendirme

if __name__ == "__main__":
    if LOSS_TYPE == "cb":
        from train_yolov8_nano_cb import train as train_cb
        train_cb(
            data=DATA_YAML,
            class_counts=CLASS_COUNTS,
            cfg=CB_CFG,
        )
    elif LOSS_TYPE == "focal":
        from train_yolov8_nano_fcl import train as train_fcl
        train_fcl(
            data=DATA_YAML,
            cfg=FCL_CFG,
        )
    elif LOSS_TYPE == "default":
        from train_yolov8_nano_default import train as train_default
        train_default(
            data=DATA_YAML,
            cfg=DEFAULT_CFG,
        )
    else:
        raise ValueError(
            f"LOSS_TYPE must be 'cb', 'focal' or 'default', got: {LOSS_TYPE!r}"
        )
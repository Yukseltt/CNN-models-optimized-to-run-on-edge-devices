# run_training.py
#
# Launcher script that starts training with CB Loss, Focal Loss, or default mode.

import sys
from pathlib import Path

# src import'lari icin proje kokunu sys.path'e ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Trainer import'lari icin mevcut dizini ekle.
sys.path.insert(0, str(Path(__file__).parent))

# Loss tipi seciciyi calistirmadan once "cb", "focal" veya "default" olarak ayarla.
LOSS_TYPE = "focal"

# Shared / Ortak
DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml"

# EKLENEN KISIM 1: Yarım kalan eğitimin last.pt dosya yolu (Dinamik olarak oluşturuldu)
RESUME_WEIGHTS = str(PROJECT_ROOT / "runs" / "yolo26n_default_augmented_2026_05_14-2" / "weights" / "last.pt")

# CB Loss configuration / CB Loss konfigurasyonu
CLASS_COUNTS = [335428, 577119, 36186]

CB_CFG = {
    "BETA":        0.99999999,
    "NAME":        "yolo26n_cb_beta_0.99999999_augmented_2026_05_14",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

# Focal Loss configuration / Focal Loss konfigurasyonu
FCL_CFG = {
    "FCL_ALPHA":   0.25,
    "FCL_GAMMA":   2.0,
    "NAME":        "yolo26n_fcl_alpha_0.25_gamma_2.0_augmented_2026_05_14",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

# Default configuration / Default konfigurasyon
DEFAULT_CFG = {
    "NAME":        "yolo26n_default_augmented_2026_05_14", 
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

# Dispatch / Yonlendirme
if __name__ == "__main__":
    if LOSS_TYPE == "cb":
        from train_yolo26_nano_cb import train as train_cb
        train_cb(
            data=DATA_YAML,
            class_counts=CLASS_COUNTS,
            cfg=CB_CFG,
        )
    elif LOSS_TYPE == "focal":
        from train_yolo26_nano_fcl import train as train_fcl
        train_fcl(
            data=DATA_YAML,
            cfg=FCL_CFG,
        )
    elif LOSS_TYPE == "default":
        from train_yolo26_nano_default import train as train_default
        train_default(
            data=DATA_YAML,
            cfg=DEFAULT_CFG,
            resume_weight_path=RESUME_WEIGHTS  # <-- EKLENEN KISIM 2: Resume yolunu buraya ekledik
        )
    else:
        raise ValueError(f"LOSS_TYPE must be 'cb', 'focal' or 'default', got: {LOSS_TYPE!r}")
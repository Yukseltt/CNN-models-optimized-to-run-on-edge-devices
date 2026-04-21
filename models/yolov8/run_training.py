"""
run_training.py
===============
Eğitimi başlatan launcher script.

KULLANIM:
    python run_training.py

Tüm parametreler burada tanımlanır.
NANO_CFG override edilmek istenmeyen değerler kaldırılabilir.
"""

import sys
from pathlib import Path

# train_yolov8_cb.py ile aynı dizinde olmalı
sys.path.insert(0, str(Path(__file__).parent))

from train_yolov8_nano_cb import train

# ──────────────────────────────────────────────────────────────
# Veri seti
# ──────────────────────────────────────────────────────────────

DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/dataset_augmented_yolo/data.yaml"      # ← data.yaml yolunu güncelle

# Her sınıfa ait eğitim örneği sayıları (data.yaml'daki sınıf
# sırasıyla eşleşmeli)
CLASS_COUNTS = [335428, 577119, 36186]

CFG_OVERRIDE = {
    "BETA":        0.99999,
    "NAME":        "yolov8n_cb_v2",
    "OUTPUT_XLSX": "training_results_2.xlsx",
}
# ──────────────────────────────────────────────────────────────
# Başlat
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        data=DATA_YAML,
        class_counts=CLASS_COUNTS,
        cfg=CFG_OVERRIDE,
    )
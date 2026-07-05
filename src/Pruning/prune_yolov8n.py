# prune_yolov8n.py
#
# YOLOv8n default augmented (mAP@0.5=0.1336, ~3M param, 176 FPS - en hizli)
# icin structured pruning + fine-tune.
#
# Kullanim:
#   python prune_yolov8n.py          # tam pipeline (sweep + prune + fine-tune)
#   python prune_yolov8n.py smoke    # hizli sanity: sadece load+prune+kaydet

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prune_lib_yolo import prune_yolo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "smoke" in sys.argv

CFG = {
    "MODEL_PATH": PROJECT_ROOT / "runs/yolov8n_default_augmented_29.04.2026/weights/best.pt",
    "DATA_YAML":  PROJECT_ROOT / "dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml",
    "OUT":        PROJECT_ROOT / "runs_pruing/yolov8n",
    "IMG":        640,
    "MIN_FLOOR":  0.10,
    "MAX_CEIL":   0.40,
    # --- fine-tune adimlari kapatildi (yorum satirina alindi) ---
    # Sadece sweep + fiziksel kesim + (kesim sonrasi) eval yapilir.
    "DO_FINETUNE": False,
    # "FINETUNE_EPOCHS": 10,
    # "BATCH":      8,
}

if SMOKE:
    CFG.update(DO_SENSITIVITY=False, DO_FINETUNE=False, SKIP_BASELINE=True, DO_ONNX=False)

if __name__ == "__main__":
    print(prune_yolo(CFG))

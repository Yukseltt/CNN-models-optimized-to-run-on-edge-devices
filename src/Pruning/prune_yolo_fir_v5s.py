# prune_yolo_fir_v5s.py
#
# YOLO-FIR v5s (benchmark'taki en iyi YOLO, mAP@0.5=0.1565) icin structured
# pruning + fine-tune.
#
# Kullanim:
#   python prune_yolo_fir_v5s.py          # tam pipeline (sweep + prune + fine-tune)
#   python prune_yolo_fir_v5s.py smoke    # hizli sanity: sadece load+prune+kaydet

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prune_lib_yolo import prune_yolo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "smoke" in sys.argv

CFG = {
    "MODEL_PATH": PROJECT_ROOT / "runs/yolo_fir_v5s_28.05.2026/weights/best.pt",
    "DATA_YAML":  PROJECT_ROOT / "dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml",
    "OUT":        PROJECT_ROOT / "runs_pruing/yolo_fir_v5s",
    "IMG":        640,
    "MIN_FLOOR":  0.10,
    "MAX_CEIL":   0.40,
    # Fine-tune ACIK olmali: YOLO structured pruning fine-tune OLMADAN cokuyor
    # (mAP ~0). BN recalibration denendi, yetmedi -> dogru cozum fine-tune.
    "DO_FINETUNE": True,
    "FINETUNE_EPOCHS": 40,
    "BATCH":      16,
    "LR0":        0.001,
}

if SMOKE:
    CFG.update(DO_SENSITIVITY=False, DO_FINETUNE=False, SKIP_BASELINE=True, DO_ONNX=False)

if __name__ == "__main__":
    print(prune_yolo(CFG))

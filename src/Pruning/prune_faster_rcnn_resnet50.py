# prune_faster_rcnn_resnet50.py
#
# Faster R-CNN ResNet50-FPN (benchmark lideri, mAP@0.5=0.3811, ~41M param)
# icin backbone-only structured pruning + fine-tune.
#
# Kullanim:
#   python prune_faster_rcnn_resnet50.py          # tam pipeline
#   python prune_faster_rcnn_resnet50.py smoke    # hizli sanity: load+prune+verify

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prune_lib_frcnn import prune_frcnn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "smoke" in sys.argv

CFG = {
    "MODEL_PATH":  PROJECT_ROOT / "runs/faster_rcnn_resnet50_28.04.2026/best.pt",
    "DATA_DIR":    PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented",
    "OUT":         PROJECT_ROOT / "runs_pruing/faster_rcnn_resnet50",
    "PRUNE_RATIO": 0.30,
    "BATCH_SIZE":  8,            # baseline val dataloader'i icin de gerekli
    # --- fine-tune adimlari kapatildi (yorum satirina alindi) ---
    "DO_FINETUNE": False,
    # "FINETUNE_EPOCHS": 12,
    # "LR0":         0.001,
}

if SMOKE:
    CFG.update(DO_FINETUNE=False, SKIP_BASELINE=True)

if __name__ == "__main__":
    print(prune_frcnn(CFG))

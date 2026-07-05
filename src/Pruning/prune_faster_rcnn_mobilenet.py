# prune_faster_rcnn_mobilenet.py
#
# Faster R-CNN MobileNetV3-Large-FPN (scratch, mAP@0.5=0.2180, ~19M param,
# edge adayi) icin backbone-only structured pruning + fine-tune.
#
# NOT: MobileNet backbone'unda depthwise/grouped conv'lar var; torch_pruning
# DepGraph bunlarin grup kisitlarini otomatik korur (round_to=1 en guvenli).
#
# Kullanim:
#   python prune_faster_rcnn_mobilenet.py          # tam pipeline
#   python prune_faster_rcnn_mobilenet.py smoke    # hizli sanity: load+prune+verify

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prune_lib_frcnn import prune_frcnn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "smoke" in sys.argv

CFG = {
    "MODEL_PATH":  PROJECT_ROOT / "runs/faster_rcnn_mobilenet_scratch_12.05.2026/best.pt",
    "DATA_DIR":    PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented",
    "OUT":         PROJECT_ROOT / "runs_pruing/faster_rcnn_mobilenet",
    "PRUNE_RATIO": 0.30,
    "BATCH_SIZE":  16,           # baseline val dataloader'i icin de gerekli
    # --- fine-tune adimlari kapatildi (yorum satirina alindi) ---
    "DO_FINETUNE": False,
    # "FINETUNE_EPOCHS": 15,
    # "LR0":         0.002,
}

if SMOKE:
    CFG.update(DO_FINETUNE=False, SKIP_BASELINE=True)

if __name__ == "__main__":
    print(prune_frcnn(CFG))

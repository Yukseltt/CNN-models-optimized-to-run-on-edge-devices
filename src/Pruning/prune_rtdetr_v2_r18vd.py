# prune_rtdetr_v2_r18vd.py
#
# RT-DETR v2 r18vd (transformer DETR, from_scratch, mAP@0.5=0.2350, ~20M param)
# icin backbone-only structured pruning + HF Trainer fine-tune.
#
# Transformer encoder/decoder (d_model=256) DOKUNULMAZ; sadece ResNet18
# backbone'u budanir (bkz. prune_lib_rtdetr.py docstring).
#
# Kullanim:
#   python prune_rtdetr_v2_r18vd.py          # tam pipeline
#   python prune_rtdetr_v2_r18vd.py smoke    # hizli sanity: load+prune+verify

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prune_lib_rtdetr import prune_rtdetr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE = "smoke" in sys.argv

CFG = {
    "MODEL_PATH":  PROJECT_ROOT / "runs/rtdetr_v2_r18vd_from_scratch_26.05.2026/best.pt",
    "DATA_DIR":    PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented",
    "OUT":         PROJECT_ROOT / "runs_pruing/rtdetr_v2_r18vd",
    "CHECKPOINT":  "PekingU/rtdetr_v2_r18vd",
    "IMAGE_SIZE":  480,
    "PRUNE_RATIO": 0.30,
    # Fine-tune ACIK olmali: RT-DETR backbone pruning fine-tune OLMADAN cokuyor
    # (transformer, pruned backbone feature'larina OOD -> mAP 0). FrozenBN oldugu
    # icin BN recalib da kurtarmaz -> dogru cozum fine-tune.
    "DO_FINETUNE": True,
    "FINETUNE_EPOCHS": 30,
    "TRAIN_BS":    16,
    "LR":          1e-4,
}

if SMOKE:
    CFG.update(DO_FINETUNE=False, SKIP_BASELINE=True)

if __name__ == "__main__":
    print(prune_rtdetr(CFG))

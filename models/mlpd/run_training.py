# run_training.py
#
# Launcher script that starts MLPD-flavored RetinaNet training.
# MLPD-flavored RetinaNet egitimini baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# Switch between configurations by changing the RUN_TYPE variable below.
# Konfigurasyonlar arasi gecis icin asagidaki RUN_TYPE'i degistir.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_mlpd import train


# Run type selector. One of:
# Run tipi seciciyi. Sunlardan biri:
#   "resnet50_coco"     - RetinaNet ResNet50+FPN, COCO pretrained (transfer)
#   "resnet50_scratch"  - RetinaNet ResNet50+FPN, backbone-only ImageNet
#   "resume"            - resume an interrupted run from last.pt

RUN_TYPE = "resnet50_coco"


# Shared / Ortak
DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_coco_dataset/dataset_augmented"


# Transfer learning configuration / Transfer learning konfigurasyonu.
#
# RetinaNet-ResNet50 is heavier than SSDLite (~38M params vs ~2.2M); the
# host is an H200 MIG 1g.35gb slice — 35GB VRAM is plenty but compute is
# limited (~1/5 of full H200), so we stay at moderate batch size.
# RetinaNet-ResNet50 SSDLite'tan daha agir (~38M vs ~2.2M); host H200 MIG
# 1g.35gb (35GB VRAM bol, compute ~1/5 H200), bu yuzden batch orta.

RESNET50_COCO_CFG = {
    "BACKBONE":     "resnet50",
    "PRETRAINED":   "coco",
    "BATCH_SIZE":   16,
    "WORKERS":      4,
    "PREFETCH":     2,
    "USE_AMP":      True,
    "AMP_DTYPE":    "bfloat16",
    "LR0":          0.005,
    "LRF":          0.01,
    "WEIGHT_DECAY": 0.0001,
    "EPOCHS":       50,
    "PATIENCE":     10,
    "NAME":         "mlpd_retinanet_resnet50_28.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Backbone-only ImageNet pretrained, head from scratch.
# Sadece backbone ImageNet pretrained, head sifirdan.
#
# Longer training / patience because detection head is random.
# Detection head rastgele oldugu icin daha uzun egitim / patience.

RESNET50_SCRATCH_CFG = {
    "BACKBONE":     "resnet50",
    "PRETRAINED":   "backbone_only",
    "BATCH_SIZE":   16,
    "WORKERS":      4,
    "PREFETCH":     2,
    "USE_AMP":      True,
    "AMP_DTYPE":    "bfloat16",
    "LR0":          0.005,
    "LRF":          0.01,
    "WEIGHT_DECAY": 0.0001,
    "EPOCHS":       100,
    "PATIENCE":     20,
    "NAME":         "mlpd_retinanet_resnet50_scratch_28.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Resume config / Resume config.
RESUME_CFG = {
    "BACKBONE":     "resnet50",
    "PRETRAINED":   "coco",
    "BATCH_SIZE":   16,
    "WORKERS":      4,
    "PREFETCH":     2,
    "USE_AMP":      True,
    "AMP_DTYPE":    "bfloat16",
    "EPOCHS":       50,
    "PATIENCE":     10,
    "NAME":         "mlpd_retinanet_resnet50_28.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
    "RESUME_FROM":  str(PROJECT_ROOT / "runs" / "mlpd_retinanet_resnet50_28.05.2026" / "last.pt"),
}


CONFIGS = {
    "resnet50_coco":    RESNET50_COCO_CFG,
    "resnet50_scratch": RESNET50_SCRATCH_CFG,
    "resume":           RESUME_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )
    train(data_dir=DATA_DIR, cfg=CONFIGS[RUN_TYPE])

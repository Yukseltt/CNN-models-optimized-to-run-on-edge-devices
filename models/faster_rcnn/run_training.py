# run_training.py
#
# Launcher script that starts Faster R-CNN training.
# Faster R-CNN egitimini baslatan launcher script.
#
# Usage / Kullanim:
#     python run_training.py
#
# Switch between configurations by changing the RUN_TYPE variable below.
# Konfigurasyonlar arasi gecis icin asagidaki RUN_TYPE degiskenini degistir.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_faster_rcnn import train


# Run type selector. One of:
# Run tipi seciciyi. Sunlardan biri:
#   "resnet50_coco"          - ResNet50 with COCO pretrained (transfer learning)
#   "mobilenet_coco"         - MobileNet with COCO pretrained (transfer learning)
#   "resnet50_scratch"       - ResNet50 backbone-only ImageNet, head from scratch
#   "mobilenet_scratch"      - MobileNet backbone-only ImageNet, head from scratch

RUN_TYPE = "mobilenet_scratch"


# Shared / Ortak

DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_coco_dataset/dataset_augmented"


# Transfer learning configurations / Transfer learning konfigurasyonlari.

RESNET50_COCO_CFG = {
    "BACKBONE":    "resnet50",
    "PRETRAINED":  "coco",
    "BATCH_SIZE":  8,
    "LR0":         0.005,
    "EPOCHS":      50,
    "PATIENCE":    10,
    "NAME":        "faster_rcnn_resnet50_28.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

MOBILENET_COCO_CFG = {
    "BACKBONE":    "mobilenet",
    "PRETRAINED":  "coco",
    "BATCH_SIZE":  16,
    "LR0":         0.01,
    "EPOCHS":      50,
    "PATIENCE":    10,
    "NAME":        "faster_rcnn_mobilenet_29.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Backbone-only ImageNet pretrained, detection head from scratch.
# Sadece backbone ImageNet pretrained, detection head sifirdan.
#
# Longer training / patience because detection layers are random and need
# more time to converge.
# Detection katmanlari rastgele oldugu icin daha uzun egitim / patience,
# convergence icin daha cok zaman gerekiyor.

RESNET50_SCRATCH_CFG = {
    "BACKBONE":    "resnet50",
    "PRETRAINED":  "backbone_only",
    "BATCH_SIZE":  8,
    "LR0":         0.005,
    "EPOCHS":      100,
    "PATIENCE":    20,
    "NAME":        "faster_rcnn_resnet50_scratch_29.04.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
    "RESUME_FROM": "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs/faster_rcnn_resnet50_scratch_29.04.2026/last.pt",
}

MOBILENET_SCRATCH_CFG = {
    "BACKBONE":    "mobilenet",
    "PRETRAINED":  "backbone_only",
    "BATCH_SIZE":  16,
    "LR0":         0.01,
    "EPOCHS":      100,
    "PATIENCE":    20,
    "NAME":        "faster_rcnn_mobilenet_scratch_12.05.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}


# Dispatch / Yonlendirme

CONFIGS = {
    "resnet50_coco":     RESNET50_COCO_CFG,
    "mobilenet_coco":    MOBILENET_COCO_CFG,
    "resnet50_scratch":  RESNET50_SCRATCH_CFG,
    "mobilenet_scratch": MOBILENET_SCRATCH_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )
    train(data_dir=DATA_DIR, cfg=CONFIGS[RUN_TYPE])
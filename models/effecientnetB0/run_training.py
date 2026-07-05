# run_training.py
#
# Launcher script that starts EfficientNet-B0 + FPN Faster R-CNN object
# detection training.
# EfficientNet-B0 + FPN Faster R-CNN obje tespiti egitimini baslatan launcher.
#
# NOT: Bu modul artik bir OBJE TESPITI detektoru egitir (eski bbox-crop
# siniflandirici launcher'i run_classifier_training.py'ye tasinmistir).
# NOTE: This module now trains an OBJECT DETECTION model (the old bbox-crop
# classifier launcher was moved to run_classifier_training.py).
#
# Usage / Kullanim:
#     python run_training.py                       # varsayilan: efficientnet_imagenet
#     python run_training.py efficientnet_imagenet
#     python run_training.py efficientnet_imagenet /yol/last.pt   # resume
#
# Switch between configurations by changing the RUN_TYPE argument.
# Konfigurasyonlar arasi gecis icin RUN_TYPE argumanini degistir.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_efficientnet_det import train


# Run type selector. One of:
# Run tipi seciciyi. Sunlardan biri:
#   "efficientnet_imagenet" - EfficientNet-B0 ImageNet1K backbone, head'ler sifirdan
#   "efficientnet_scratch"  - her sey rastgele (backbone dahil)
RUN_TYPE = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_imagenet"


# Shared / Ortak

# restratified_coco: yeniden stratifiye edilmis dataset (train 35190 / val 4470 / test 4480).
# Eski dataset: dataset/2x_augmented_coco_dataset/dataset_augmented
DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/restratified_coco"

# Tum yeni egitimler runs_new/ altina kaydedilir.
RUNS_NEW = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs_new"


# Transfer learning (ImageNet backbone, detection head'ler sifirdan).
# Transfer learning (ImageNet backbone, detection heads from scratch).
#
# EfficientNet-B0 backbone ImageNet1K agirliklariyla baslar; FPN/RPN/box head
# rastgele oldugundan convergence icin daha uzun egitim/patience kullaniriz.
# EfficientNet-B0 backbone starts from ImageNet1K; FPN/RPN/box head are random,
# so we use longer training/patience for convergence.

EFFICIENTNET_IMAGENET_CFG = {
    "PRETRAINED":   "imagenet",
    "TRAINABLE_BACKBONE_LAYERS": None,   # tum backbone egitilebilir
    "BATCH_SIZE":   8,
    "LR0":          0.005,
    "EPOCHS":       150,
    "PATIENCE":     20,
    "MIN_SIZE":     640,
    "MAX_SIZE":     640,
    "PROJECT":      RUNS_NEW,
    "NAME":         "efficientnet_b0_fpn_fasterrcnn_restratified_20.06.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Everything random (backbone included), heads from scratch.
# Her sey rastgele (backbone dahil), head'ler sifirdan.

EFFICIENTNET_SCRATCH_CFG = {
    "PRETRAINED":   None,
    "TRAINABLE_BACKBONE_LAYERS": None,
    "BATCH_SIZE":   8,
    "LR0":          0.005,
    "EPOCHS":       150,
    "PATIENCE":     20,
    "MIN_SIZE":     640,
    "MAX_SIZE":     640,
    "PROJECT":      RUNS_NEW,
    "NAME":         "efficientnet_b0_fpn_fasterrcnn_scratch_restratified_20.06.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Dispatch / Yonlendirme

CONFIGS = {
    "efficientnet_imagenet": EFFICIENTNET_IMAGENET_CFG,
    "efficientnet_scratch":  EFFICIENTNET_SCRATCH_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )
    cfg = dict(CONFIGS[RUN_TYPE])
    # Opsiyonel resume: `python run_training.py efficientnet_imagenet /yol/last.pt`
    # 2. arguman verilirse o checkpoint'ten devam eder; cikti o dizine yazilir.
    if len(sys.argv) > 2:
        cfg["RESUME_FROM"] = sys.argv[2]
    train(data_dir=DATA_DIR, cfg=cfg)

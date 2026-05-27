# run_training.py
#
# Launcher script that starts SSD (SSDLite320 + MobileNetV3-Large) training.
# SSD (SSDLite320 + MobileNetV3-Large) egitimini baslatan launcher script.
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

from train_ssd import train


# Run type selector. One of:
# Run tipi seciciyi. Sunlardan biri:
#   "mobilenet_coco"     - SSDLite + MobileNetV3 COCO pretrained (transfer learning)
#   "mobilenet_scratch"  - SSDLite + MobileNetV3 backbone-only ImageNet, head from scratch

RUN_TYPE = "mobilenet_coco"


# Shared / Ortak

DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_coco_dataset/dataset_augmented"


# Transfer learning configuration / Transfer learning konfigurasyonu.
#
# Lower LR than scratch because head starts from COCO weights and only needs
# light fine-tuning to remap to our 3 classes.
# Head COCO agirliklarindan basladigi icin sadece hafif fine-tuning gerekiyor,
# bu yuzden scratch'e gore daha dusuk LR.

MOBILENET_COCO_CFG = {
    "BACKBONE":     "mobilenet",
    "PRETRAINED":   "coco",
    # H200 MIG 1g.35gb has plenty of VRAM for SSDLite (~2.2M params, 320x320),
    # so we push batch size for higher throughput. CPU is shared on the host,
    # so workers stay below physical cores.
    # H200 MIG 1g.35gb'de SSDLite icin VRAM bol; throughput icin batch yukari.
    # Host CPU paylasimli, workers fiziksel core sayisinin altinda.
    "BATCH_SIZE":   64,
    "WORKERS":      4,
    "PREFETCH":     2,
    "USE_AMP":      True,
    "AMP_DTYPE":    "bfloat16",
    "LR0":          0.01,
    "LRF":          0.01,
    "WEIGHT_DECAY": 0.00004,
    "EPOCHS":       300,
    "PATIENCE":     20,
    "NAME":         "ssdlite_mobilenet_27.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Backbone-only ImageNet pretrained, detection head from scratch.
# Sadece backbone ImageNet pretrained, detection head sifirdan.
#
# Longer training / patience because detection layers are random and need
# more time to converge.
# Detection katmanlari rastgele oldugu icin daha uzun egitim / patience.

MOBILENET_SCRATCH_CFG = {
    "BACKBONE":     "mobilenet",
    "PRETRAINED":   "backbone_only",
    "BATCH_SIZE":   64,
    "WORKERS":      4,
    "PREFETCH":     2,
    "USE_AMP":      True,
    "AMP_DTYPE":    "bfloat16",
    "LR0":          0.05,
    "LRF":          0.01,
    "WEIGHT_DECAY": 0.00004,
    "EPOCHS":       100,
    "PATIENCE":     20,
    "NAME":         "ssdlite_mobilenet_scratch_27.05.2026",
    "OUTPUT_XLSX":  "training_metrics.xlsx",
}


# Dispatch / Yonlendirme

CONFIGS = {
    "mobilenet_coco":    MOBILENET_COCO_CFG,
    "mobilenet_scratch": MOBILENET_SCRATCH_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )
    train(data_dir=DATA_DIR, cfg=CONFIGS[RUN_TYPE])

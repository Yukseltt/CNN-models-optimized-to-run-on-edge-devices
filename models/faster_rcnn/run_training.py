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

# Komut satirindan secilebilir: `python run_training.py mobilenet_coco`
# Arguman verilmezse asagidaki varsayilan kullanilir.
RUN_TYPE = sys.argv[1] if len(sys.argv) > 1 else "resnet50_coco"


# Shared / Ortak

# restratified_coco: yeniden stratifiye edilmis dataset (train 35190 / val 4470 / test 4480).
# Eski dataset: dataset/2x_augmented_coco_dataset/dataset_augmented
DATA_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/restratified_coco"

# Tum yeni egitimler runs_new/ altina kaydedilir.
RUNS_NEW = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs_new"


# Transfer learning configurations / Transfer learning konfigurasyonlari.

RESNET50_COCO_CFG = {
    "BACKBONE":    "resnet50",
    "PRETRAINED":  "coco",
    "BATCH_SIZE":  8,
    "LR0":         0.005,
    "EPOCHS":      150,
    "PATIENCE":    15,
    "PROJECT":     RUNS_NEW,
    "NAME":        "faster_rcnn_resnet50_restratified_17.06.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

MOBILENET_COCO_CFG = {
    "BACKBONE":    "mobilenet",
    "PRETRAINED":  "coco",
    "BATCH_SIZE":  16,
    "LR0":         0.01,
    "EPOCHS":      150,
    "PATIENCE":    15,
    "PROJECT":     RUNS_NEW,
    "NAME":        "faster_rcnn_mobilenet_restratified_17.06.2026",
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
    "PROJECT":     RUNS_NEW,
    "NAME":        "faster_rcnn_resnet50_scratch_restratified_17.06.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
}

MOBILENET_SCRATCH_CFG = {
    "BACKBONE":    "mobilenet",
    "PRETRAINED":  "backbone_only",
    "BATCH_SIZE":  16,
    "LR0":         0.01,
    "EPOCHS":      150,
    "PATIENCE":    15,
    "PROJECT":     RUNS_NEW,
    "NAME":        "faster_rcnn_mobilenet_scratch_restratified_17.06.2026",
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
    cfg = dict(CONFIGS[RUN_TYPE])
    # Opsiyonel resume: `python run_training.py mobilenet_coco /yol/last.pt`
    # 2. arguman verilirse o checkpoint'ten devam eder; cikti o dizine yazilir.
    if len(sys.argv) > 2:
        cfg["RESUME_FROM"] = sys.argv[2]
    train(data_dir=DATA_DIR, cfg=cfg)
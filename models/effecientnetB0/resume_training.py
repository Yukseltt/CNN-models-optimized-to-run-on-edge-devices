# resume_training.py
#
# Resume launcher for EfficientNet-B0 + FPN Faster R-CNN object detection.
# EfficientNet-B0 + FPN Faster R-CNN obje tespiti icin devam ettirici.
#
# Yarida kesilen son egitimi run dizinindeki last.pt'den otomatik bulup devam
# ettirir; ayni run klasorune yazmaya devam eder, training_metrics.xlsx'i
# gunceller. Hiperparametreler run_training.py'den okunur (tek kaynak).
# Resumes the last interrupted run by auto-locating last.pt in the run dir;
# keeps writing to the same folder and updates training_metrics.xlsx.
# Hyperparameters are read from run_training.py (single source of truth).
#
# NOT: Eski bbox-crop siniflandirici resume launcher'i
# resume_classifier_training.py'dedir.
# NOTE: The old bbox-crop classifier resume launcher is in
# resume_classifier_training.py.
#
# Usage / Kullanim:
#     python resume_training.py                                 # varsayilan config'in last.pt'si
#     python resume_training.py efficientnet_imagenet           # belirli config
#     python resume_training.py efficientnet_imagenet /yol/last.pt   # acik checkpoint

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from train_efficientnet_det import train
# run_training.py ile ayni config/dataset; tek kaynak (config drift olmasin).
# Same config/dataset as run_training.py; single source of truth (no drift).
from run_training import CONFIGS, DATA_DIR


RUN_TYPE = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_imagenet"


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(
            f"RUN_TYPE must be one of {list(CONFIGS.keys())}, got: {RUN_TYPE!r}"
        )

    cfg = dict(CONFIGS[RUN_TYPE])

    # Checkpoint: 2. arguman verilirse onu kullan; yoksa run dizinindeki last.pt.
    # Checkpoint: use the 2nd arg if given, else the run dir's last.pt.
    if len(sys.argv) > 2:
        ckpt = Path(sys.argv[2])
    else:
        ckpt = Path(cfg["PROJECT"]) / cfg["NAME"] / "last.pt"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {ckpt}\n"
            f"Egitim dizini farkliysa (or. NAME2) checkpoint yolunu 2. arguman "
            f"olarak ver: python resume_training.py {RUN_TYPE} /yol/last.pt"
        )

    cfg["RESUME_FROM"] = str(ckpt)
    print(f"[EFFB0-DET] Resuming '{RUN_TYPE}' from {ckpt}")
    train(data_dir=DATA_DIR, cfg=cfg)

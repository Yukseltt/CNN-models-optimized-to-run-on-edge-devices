# run_training.py
# Launcher: python run_training.py default|cb|focal

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

LOSS_TYPE = sys.argv[1] if len(sys.argv) > 1 else "default"
RESUME_FROM = sys.argv[2] if len(sys.argv) > 2 else None

DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/restratified_yolo/data.yaml"
RUNS_NEW  = str(PROJECT_ROOT / "runs_new")

# CLASS_COUNTS: restratified train set (person / car / other_vehicle)
CLASS_COUNTS = [207400, 319437, 27099]

CB_CFG = {
    "BETA":        0.99999999,
    "NAME":        "yolo11n_cb_beta_0.99999999_restratified_01.07.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
    "PROJECT":     RUNS_NEW,
}

FCL_CFG = {
    "FCL_ALPHA":   0.25,
    "FCL_GAMMA":   2.0,
    "NAME":        "yolo11n_fcl_alpha_0.25_gamma_2.0_restratified_01.07.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
    "PROJECT":     RUNS_NEW,
}

DEFAULT_CFG = {
    "NAME":        "yolo11n_default_restratified_01.07.2026",
    "OUTPUT_XLSX": "training_metrics.xlsx",
    "PROJECT":     RUNS_NEW,
}

if __name__ == "__main__":
    if LOSS_TYPE == "cb":
        from train_yolo11n_cb import train as train_cb
        train_cb(data=DATA_YAML, class_counts=CLASS_COUNTS, cfg=CB_CFG)
    elif LOSS_TYPE == "focal":
        from train_yolo11n_fcl import train as train_fcl
        train_fcl(data=DATA_YAML, cfg=FCL_CFG)
    elif LOSS_TYPE == "default":
        from train_yolo11n_default import train as train_default
        train_default(data=DATA_YAML, cfg=DEFAULT_CFG, resume_weight_path=RESUME_FROM)
    else:
        raise ValueError(f"LOSS_TYPE must be 'cb', 'focal' or 'default', got: {LOSS_TYPE!r}")

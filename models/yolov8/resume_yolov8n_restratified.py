# resume_yolov8n_restratified.py
#
# Resume the interrupted YOLOv8-nano training on the restratified dataset.
# Restratified dataset uzerindeki yarida kalan YOLOv8-nano egitimini devam ettirir.
#
# Egitim epoch 33/300'te yarida kaldi. last.pt tam egitim state'ini
# (optimizer, ema, scaler, train_args) icerdigi icin Ultralytics resume=True
# ile kaldigi yerden (epoch 34) devam eder. Ayni save_dir'e yazar, results.csv
# ve training_metrics.xlsx ayni dosyalara eklenmeye devam eder.
#
# Usage / Kullanim:
#     python resume_yolov8n_restratified.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from train_yolov8_nano_default import DefaultTrainingCallbacks, read_class_names


# Yarida kalan run'in dizini ve son checkpoint'i.
RUN_DIR  = PROJECT_ROOT / "runs_new" / "yolov8n_default_restratified_14.06.2026"
LAST_PT  = RUN_DIR / "weights" / "last.pt"

# Excel/per-class loglamanin ayni dosyada devami icin orijinal data.yaml.
DATA_YAML   = str(PROJECT_ROOT / "dataset" / "restratified_yolo" / "data.yaml")
OUTPUT_XLSX = "training_metrics.xlsx"


def main() -> None:
    if not LAST_PT.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {LAST_PT}")

    class_names = read_class_names(DATA_YAML)

    LOGGER.info(f"[Resume] Checkpoint: {LAST_PT}")
    LOGGER.info(f"[Resume] Classes: {class_names}")

    # last.pt'den yukle. resume=True tum egitim arg'larini checkpoint'ten okur.
    model = YOLO(str(LAST_PT))

    # Metrik loglama callback'lerini yeniden bagla. on_pretrain_routine_end
    # mevcut xlsx'i overwrite etmez (exists() kontrolu var), satir eklemeye devam eder.
    cbs = DefaultTrainingCallbacks(
        output_xlsx_name=OUTPUT_XLSX,
        class_names=class_names,
    )
    model.add_callback("on_pretrain_routine_end", cbs.on_pretrain_routine_end)
    model.add_callback("on_fit_epoch_end",        cbs.on_fit_epoch_end)
    model.add_callback("on_train_end",            cbs.on_train_end)

    # Kaldigi yerden devam et. Diger arg'lar checkpoint'ten gelir.
    model.train(resume=True)


if __name__ == "__main__":
    main()

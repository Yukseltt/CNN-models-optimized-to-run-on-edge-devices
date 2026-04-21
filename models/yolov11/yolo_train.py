import random
import shutil
import cv2
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
# ──────────────────────────────────────────────
# DESTEKLENEN MODELLER
# ──────────────────────────────────────────────
SUPPORTED_MODELS = {
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv9
    "yolov9c": "yolov9c.pt",
    "yolov9e": "yolov9e.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    # YOLOv11
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO12 (ultralytics>=8.3.x gerektirir)
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    # RT-DETR
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}

# ──────────────────────────────────────────────
# VARSAYILAN KONFİGÜRASYON
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Yollar
    "ROOT"               : Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection"),
    "DATASET_ROOT"       : Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/thermal_yolo"),
    "EXP_NAME"           : "thermal_yolo",

    # Model — buradan değiştir: "yolo11n", "yolo12s", "yolov8m" vs.
    "MODEL_KEY"          : "yolo11n",

    # Eğitim
    "EPOCHS"             : 1000,
    "BATCH_SIZE"         : 8,
    "IMG_SIZE"           : 640,
    "DEVICE"             : 0,       # 0 = GPU, "cpu" = CPU
    "RESUME"             : False,
    "PATIENCE"           : 25,
    "WORKERS"            : 2,
    "CACHE"              : False,

    # Optimizer
    "OPTIMIZER"          : "SGD",
    "LR0"                : 0.001,
    "LRF"                : 0.01,
    "MOMENTUM"           : 0.937,
    "WEIGHT_DECAY"       : 0.0005,
    "WARMUP_EPOCHS"      : 3,
    "AMP"                : True,

    # Augmentation
    "AUGMENT_TRAIN"      : False,
    "AUGMENT_MULTIPLIER" : 2,
    "HSV_H"              : 0.015,
    "HSV_S"              : 0.7,
    "HSV_V"              : 0.4,
    "FLIPUD"             : 0.0,
    "FLIPLR"             : 0.5,
    "DEGREES"            : 0.0,
    "TRANSLATE"          : 0.1,
    "SCALE"              : 0.5,
    "PERSPECTIVE"        : 0.0,
    "ERASING"            : 0.4,
    "MOSAIC"             : 1.0,
    "MIXUP"              : 0.0,
}


def _resolve_config(cfg: dict) -> dict:
    """Varsayılan config üzerine kullanıcı değerlerini uygular."""
    import re, json as _json
    c = {**DEFAULT_CONFIG, **cfg}
    c["ROOT"]         = Path(c["ROOT"])
    c["DATASET_ROOT"] = Path(c["DATASET_ROOT"])
    c["PROJECT_DIR"]  = str(c["ROOT"] / "runs")
    c["SAVE_DIR"]     = c["ROOT"] / "saved_models"
    c["yaml_path"]    = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/dataset_augmented_yolo/data.yaml"

    model_key = c["MODEL_KEY"].lower()
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"[HATA] '{model_key}' desteklenmiyor.\n"
            f"Desteklenen modeller: {list(SUPPORTED_MODELS.keys())}"
        )
    c["MODEL_WEIGHTS"] = SUPPORTED_MODELS[model_key]

    if "rtdetr" in model_key:
        c["MODEL_TYPE"] = "rtdetr"
        c["MODEL_SIZE"] = model_key.split("-")[-1]
    else:
        m = re.match(r"(yolo(?:v\d+|\d+))([a-z]+)", model_key)
        if m:
            c["MODEL_TYPE"] = m.group(1)
            c["MODEL_SIZE"] = m.group(2)
        else:
            c["MODEL_TYPE"] = model_key
            c["MODEL_SIZE"] = ""

    # ── config.json'dan model boyutuna göre parametreleri yükle ──
    SIZE_MAP = {"n": "nano", "s": "small"}
    size_key = SIZE_MAP.get(c["MODEL_SIZE"])
    if size_key:
        config_path = c["ROOT"] / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                ext = _json.load(f).get(size_key, {})
            # config.json değerleri DEFAULT üzerine yazılır,
            # ama kullanıcının cfg ile verdiği değerler en öncelikli kalır
            for k, v in ext.items():
                if k not in cfg:
                    c[k] = v
            print(f"[INFO]  config.json → '{size_key}' profili yüklendi")

    return c


# ──────────────────────────────────────────────
# EĞİTİM
# ──────────────────────────────────────────────
def train(cfg: dict = None):
    """
    Modeli eğitir.

    Kullanım:
        from train_yolo import train
        model, c = train()                           # varsayılan config
        model, c = train({"MODEL_KEY": "yolo12s"})  # farklı model
        model, c = train({"EPOCHS": 50, "BATCH_SIZE": 8})
    """
    c = _resolve_config(cfg or {})

    print("=" * 70)
    print(f"""  Model        : {c['MODEL_WEIGHTS']}
  Dataset      : {c['yaml_path']}
  Epochs       : {c['EPOCHS']}
  Batch        : {c['BATCH_SIZE']}
  Img Size     : {c['IMG_SIZE']}
  Device       : {c['DEVICE']}
  Resume       : {c['RESUME']}
  Patience     : {c['PATIENCE']}
  Optimizer    : {c['OPTIMIZER']}
  LR0          : {c['LR0']}
  LRF          : {c['LRF']}
  Momentum     : {c['MOMENTUM']}
  Weight Decay : {c['WEIGHT_DECAY']}
  Warmup Epochs: {c['WARMUP_EPOCHS']}
  AMP          : {c['AMP']}
  HSV_H        : {c['HSV_H']}
  HSV_S        : {c['HSV_S']}
  HSV_V        : {c['HSV_V']}
  FlipUD       : {c['FLIPUD']}
  FlipLR       : {c['FLIPLR']}
  Degrees      : {c['DEGREES']}
  Translate    : {c['TRANSLATE']}
  Scale        : {c['SCALE']}
  Perspective  : {c['PERSPECTIVE']}
  Erasing      : {c['ERASING']}
  Mosaic       : {c['MOSAIC']}
  Mixup        : {c['MIXUP']}
  Cache        : {c['CACHE']}
  Workers      : {c['WORKERS']}
  Augmentation : {"Albumentations (CPU) — multiplier x" + str(c['AUGMENT_MULTIPLIER']) if c['AUGMENT_TRAIN'] else "YOLO GPU"}""")
    print("=" * 70)

    # Resume kontrolü
    resume_path = None
    if c["RESUME"]:
        weights_dir  = Path(c["PROJECT_DIR"]) / c["EXP_NAME"] / "weights"
        last_pt_path = weights_dir / "last.pt"
        best_pt_path = weights_dir / "best.pt"
        if last_pt_path.exists():
            resume_path = str(last_pt_path)
            print(f"[INFO]  Eğitime '{resume_path}' dosyasından devam ediliyor.")
        elif best_pt_path.exists():
            resume_path = str(best_pt_path)
            print(f"[INFO]  'last.pt' bulunamadı, '{resume_path}' dosyasından devam ediliyor.")
        else:
            print(f"[WARN]  Ağırlık bulunamadı. '{c['MODEL_WEIGHTS']}' ile baştan başlanıyor.")

    model = YOLO(resume_path if resume_path else c["MODEL_WEIGHTS"])

    model.train(
        data            = str(c["yaml_path"]),
        epochs          = c["EPOCHS"],
        imgsz           = c["IMG_SIZE"],
        batch           = c["BATCH_SIZE"],
        workers         = c["WORKERS"],
        device          = c["DEVICE"],
        project         = c["PROJECT_DIR"],
        name            = c["EXP_NAME"],
        exist_ok        = True,
        save            = True,
        plots           = True,
        patience        = c["PATIENCE"],
        cache           = c["CACHE"],
        mosaic          = c["MOSAIC"],
        mixup           = c["MIXUP"],
        copy_paste      = 0.0,
        degrees         = c["DEGREES"],
        translate       = c["TRANSLATE"],
        scale           = c["SCALE"],
        perspective     = c["PERSPECTIVE"],
        erasing         = c["ERASING"],
        fliplr          = c["FLIPLR"],
        flipud          = c["FLIPUD"],
        hsv_h           = c["HSV_H"],
        hsv_s           = c["HSV_S"],
        hsv_v           = c["HSV_V"],
        optimizer       = c["OPTIMIZER"],
        lr0             = c["LR0"],
        lrf             = c["LRF"],
        momentum        = c["MOMENTUM"],
        warmup_epochs   = c["WARMUP_EPOCHS"],
        weight_decay    = c["WEIGHT_DECAY"],
        label_smoothing = 0,
        amp             = c["AMP"],
        verbose         = True,
    )

    print("\n[OK]    Eğitim tamamlandı")
    return model, c


# ──────────────────────────────────────────────
# GRAFİKLER
# ──────────────────────────────────────────────
def plot_metrics(cfg: dict = None):
    """
    Kullanım:
        from train_yolo import plot_metrics
        plot_metrics()
        plot_metrics({"MODEL_KEY": "yolo12s", "EXP_NAME": "thermal_yolo"})
    """
    c = _resolve_config(cfg or {})
    results_csv = Path(c["PROJECT_DIR"]) / c["EXP_NAME"] / "results.csv"

    if not results_csv.exists():
        print(f"[WARN]  results.csv bulunamadı: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{c['MODEL_TYPE'].upper()} {c['MODEL_SIZE'].capitalize()} — Eğitim Metrikleri",
        fontsize=15, fontweight="bold"
    )

    metric_plots = [
        ("train/box_loss",       "val/box_loss",        "Box Loss",   axes[0, 0], "Train",                "Validation"),
        ("train/cls_loss",       "val/cls_loss",        "Class Loss", axes[0, 1], "Train",                "Validation"),
        ("train/dfl_loss",       "val/dfl_loss",        "DFL Loss",   axes[0, 2], "Train",                "Validation"),
        ("metrics/precision(B)", None,                  "Precision",  axes[1, 0], "Validation Precision", None),
        ("metrics/recall(B)",    None,                  "Recall",     axes[1, 1], "Validation Recall",    None),
        ("metrics/mAP50(B)",     "metrics/mAP50-95(B)", "mAP",        axes[1, 2], "mAP@0.5",             "mAP@0.5:0.95"),
    ]

    for train_col, val_col, title, ax, train_label, val_label in metric_plots:
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], label=train_label, color="steelblue", linewidth=2)
        if val_col and val_col in df.columns:
            ax.plot(df["epoch"], df[val_col],   label=val_label,   color="tomato",    linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = c["ROOT"] / "training_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[OK]    Grafik kaydedildi: {out}")


# ──────────────────────────────────────────────
# VAL / TEST DEĞERLENDİRME
# ──────────────────────────────────────────────
def evaluate(cfg: dict = None):
    """
    Kullanım:
        from train_yolo import evaluate
        val_r, test_r, model_eval = evaluate()
        val_r, test_r, model_eval = evaluate({"MODEL_KEY": "yolo12s"})
    """
    c = _resolve_config(cfg or {})
    best_weights = Path(c["PROJECT_DIR"]) / c["EXP_NAME"] / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"[WARN]  best.pt bulunamadı: {best_weights}")
        return None, None, None

    print(f"[INFO]  En iyi ağırlıklar: {best_weights}")
    model_eval = YOLO(str(best_weights))

    print("\n[INFO]  Validasyon Seti")
    val_results = model_eval.val(
        data=str(c["yaml_path"]), split="val",
        imgsz=c["IMG_SIZE"], device=c["DEVICE"], verbose=True
    )

    print("\n[INFO]  Test Seti")
    test_results = model_eval.val(
        data=str(c["yaml_path"]), split="test",
        imgsz=c["IMG_SIZE"], device=c["DEVICE"], verbose=True
    )

    print(f"""
SONUÇLAR
  VAL  mAP@0.5      : {val_results.box.map50:.4f}
  VAL  mAP@0.5:0.95 : {val_results.box.map:.4f}
  TEST mAP@0.5      : {test_results.box.map50:.4f}
  TEST mAP@0.5:0.95 : {test_results.box.map:.4f}
""")
    return val_results, test_results, model_eval


# ──────────────────────────────────────────────
# TAHMİN GÖRSELLEŞTİRME
# ──────────────────────────────────────────────
def visualize_predictions(model, image_dir, n=8, conf=0.25, iou=0.5, cfg: dict = None):
    """
    Kullanım:
        from train_yolo import visualize_predictions
        visualize_predictions(model_eval, "dataset/.../test/images", n=8, conf=0.3)
    """
    c = _resolve_config(cfg or {})
    img_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    if not img_paths:
        print(f"[WARN]  Görüntü bulunamadı: {image_dir}")
        return

    samples = random.sample(img_paths, min(n, len(img_paths)))
    cols    = 4
    rows    = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, img_path in enumerate(samples):
        img      = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        results  = model.predict(img, conf=conf, iou=iou, verbose=False)
        res_plot = results[0].plot()
        res_rgb  = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        axes[i].imshow(res_rgb)
        axes[i].set_title(img_path.name, fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Tahminler — conf≥{conf}  iou={iou}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = c["ROOT"] / "predictions.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[OK]    Tahmin görseli kaydedildi: {out}")


# ──────────────────────────────────────────────
# YEREL DISKE KAYDET
# ──────────────────────────────────────────────
def save_results(val_results=None, test_results=None, cfg: dict = None):
    """
    Kullanım:
        from train_yolo import save_results
        save_results(val_r, test_r)
        save_results(val_r, test_r, {"MODEL_KEY": "yolo12s"})
    """
    c = _resolve_config(cfg or {})
    save_dir = c["SAVE_DIR"] / c["EXP_NAME"]
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_dir  = Path(c["PROJECT_DIR"]) / c["EXP_NAME"] / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    if best_weights.exists():
        shutil.copy2(str(best_weights), str(save_dir / "best.pt"))
    if last_weights.exists():
        shutil.copy2(str(last_weights), str(save_dir / "last.pt"))

    for png in ["training_curves.png", "predictions.png", "augmentation_preview.png"]:
        src = c["ROOT"] / png
        if src.exists():
            shutil.copy2(str(src), str(save_dir / png))

    if val_results and test_results:
        metrics_path     = save_dir / "training_metrics.xlsx"
        current_run_data = {
            "Timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Experiment_Name":    c["EXP_NAME"],
            "Model_Key":          c["MODEL_KEY"],
            "Model_Weights":      c["MODEL_WEIGHTS"],
            "Augment_Train":      c["AUGMENT_TRAIN"],
            "Augment_Multiplier": c["AUGMENT_MULTIPLIER"],
            "Epochs":             c["EPOCHS"],
            "Batch_Size":         c["BATCH_SIZE"],
            "Img_Size":           c["IMG_SIZE"],
            "Patience":           c["PATIENCE"],
            "Optimizer":          c["OPTIMIZER"],
            "LR0":                c["LR0"],
            "LRF":                c["LRF"],
            "Momentum":           c["MOMENTUM"],
            "Weight_Decay":       c["WEIGHT_DECAY"],
            "Warmup_Epochs":      c["WARMUP_EPOCHS"],
            "AMP":                c["AMP"],
            "HSV_H":              c["HSV_H"],
            "HSV_S":              c["HSV_S"],
            "HSV_V":              c["HSV_V"],
            "FlipUD":             c["FLIPUD"],
            "FlipLR":             c["FLIPLR"],
            "Degrees":            c["DEGREES"],
            "Translate":          c["TRANSLATE"],
            "Scale":              c["SCALE"],
            "Perspective":        c["PERSPECTIVE"],
            "Erasing":            c["ERASING"],
            "Mosaic":             c["MOSAIC"],
            "Mixup":              c["MIXUP"],
            "Cache":              c["CACHE"],
            "Workers":            c["WORKERS"],
            "Val_mAP50":          val_results.box.map50,
            "Val_mAP50-95":       val_results.box.map,
            "Val_Precision":      val_results.box.p,
            "Val_Recall":         val_results.box.r,
            "Test_mAP50":         test_results.box.map50,
            "Test_mAP50-95":      test_results.box.map,
            "Test_Precision":     test_results.box.p,
            "Test_Recall":        test_results.box.r,
        }
        current_df = pd.DataFrame([current_run_data])
        if metrics_path.exists():
            existing_df = pd.read_excel(metrics_path, engine="openpyxl")
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
        combined_df.to_excel(metrics_path, index=False, engine="openpyxl")

    print(f"""
[OK]  Sonuçlar kaydedildi:
  {save_dir}
  ├── best.pt
  ├── last.pt
  ├── training_curves.png
  ├── predictions.png
  └── training_metrics.xlsx
""")


# ──────────────────────────────────────────────
# ANA AKIŞ (doğrudan çalıştırılınca)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ── MODEL SEÇ ──────────────────────────────
    # YOLOv8 : "yolov8n"  "yolov8s"  "yolov8m"  "yolov8l"  "yolov8x"
    # YOLOv11: "yolo11n"  "yolo11s"  "yolo11m"  "yolo11l"  "yolo11x"
    # YOLOv12: "yolo12n"  "yolo12s"  "yolo12m"  "yolo12l"  "yolo12x"
    # RT-DETR: "rtdetr-l" "rtdetr-x"
    # ───────────────────────────────────────────
    cfg = {
        "MODEL_KEY": "yolo11n",   # <-- buradan değiştir
    }

    model, c   = train(cfg)
    plot_metrics(cfg)
    val_r, test_r, model_eval = evaluate(cfg)
    visualize_predictions(
        model_eval,
        c["DATASET_ROOT"] / "images" / "test",
        n=8, conf=0.3, cfg=cfg
    )
    save_results(val_r, test_r, cfg)
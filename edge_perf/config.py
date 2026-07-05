# config.py
#
# edge_perf paketinin TEK kayit defteri (registry).
#   - 5 model: normal (duz) + pruned agirlik yollari, tip, imgsz, nominal param.
#   - 3 varyant: normal / pruned / quantized  (her platformda farkli precision'a denk).
#   - Platforma gore (server / jetson / rpi5_hailo) hangi varyantin hangi precision
#     ve backend ile kosacaginin matrisi.
#
# main.py burayi okur; baska hicbir yerde model yolu / platform mantigi yoktur.

from pathlib import Path

# edge_perf/ -> proje koku iki ust.
PKG_ROOT     = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent
# Tum agirlik/artifact'lar burada, model basina alt klasor:
#   models/<key>/{base,pruned}.pt  {base,pruned}.onnx  {base,pruned}_int8.onnx
#               {base,pruned}_<prec>.engine   {base,pruned}.hef
# Bu klasor tek-paket tasinabilir -> Raspberry'ye/Jetson'a kopyalayip calistir.
MODELS_DIR   = PKG_ROOT / "models"
WEIGHTS_DIR  = MODELS_DIR          # geriye-uyum: eski isim hala models/'e isaret eder
RESULTS_DIR  = PKG_ROOT / "results"

# Dosya adi etiketi: ic "structure" (normal/pruned) -> disk adi (base/pruned).
STRUCT_LABEL = {"normal": "base", "pruned": "pruned"}

# ---- Test seti (mAP icin); cihazda da bu yollar gecerli olmali ----
TEST_IMAGES = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/images"
TEST_GT     = PROJECT_ROOT / "dataset/2x_augmented_coco_dataset/dataset_augmented/test/_annotations.coco.json"

# GT kategori sirasi (frcnn/rtdetr egitim sirasi). YOLO model.names bu projede gt ile
# TUTARSIZ -> mAP eslemesi POZISYON-tabanli yapilir (model_class_names verilmez).
CLASS_NAMES = ["Person", "Car", "OtherVehicle"]

# ---- Olcum parametreleri ----
WARMUP_RUNS    = 20
MEASURE_RUNS   = 100
MAP_MAX_IMAGES = None    # None -> tum test seti; int -> hizli alt-kume (smoke icin 200 gibi)

# RT-DETR v2 HF config/image_processor kaynagi (normal agirligi rebuild icin).
RTDETR_HF_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"


# =====================================================================
# Model kayit defteri
# =====================================================================
# Her model:
#   key         : kisa benzersiz ad (cikti/dosya isimlerinde kullanilir)
#   type        : "yolo" | "frcnn" | "rtdetr"  (yukleme + decode bunu izler)
#   imgsz       : inference girdi boyutu
#   params_m    : nominal param (M) — torch backend gercek degeri runtime'da hesaplar,
#                 cihaz backend'leri (engine/hef) bu nominali raporlar
#   normal_pt   : duz (unpruned) agirlik aday yollari (ilk var olan secilir)
#   pruned_pt   : pruned agirlik aday yollari
#
# Aday liste: proje run dizinleri dagilik; ilk mevcut olan kullanilir.
MODELS = [
    {
        "key": "yolov8n",
        "type": "yolo",
        "imgsz": 640,
        "params_m": 3.01,
        "params_m_pruned": 2.73,
        "normal_pt": [
            PROJECT_ROOT / "runs/yolov8n_default_augmented_29.04.2026/weights/best.pt",
        ],
        # DIKKAT: yolov8n pruned cikti dizini (runs_pruing/yolov8n) su an BOS.
        # prune_yolov8n.py DO_FINETUNE=False ile kosmus; finetuned pruned agirligi
        # YOK. Asagidaki yollar model'e OZGU tutulur (paylasimli runs_pruing/finetune
        # yolo_fir'e ait -> yanlislikla onu almasin diye eklenmedi). Bulunmazsa SKIP.
        "pruned_pt": [
            PROJECT_ROOT / "runs_pruing/yolov8n/finetune/weights/best.pt",
            PROJECT_ROOT / "runs_pruing/yolov8n/pruned_final.pt",
        ],
    },
    {
        "key": "yolo_fir_v5s",
        "type": "yolo",
        "imgsz": 640,
        "params_m": 9.12,
        "params_m_pruned": 8.55,
        "normal_pt": [
            PROJECT_ROOT / "runs/yolo_fir_v5s_28.05.2026/weights/best.pt",
        ],
        # runs_pruing/finetune/ = yolo_fir pruned+finetune (10 epoch; args.yaml dogrular).
        # runs_pruing/yolo_fir_v5s/pruned_final.pt = FINETUNE'SUZ (mAP 0.569->0.0005, KIRIK)
        # -> once finetuned olani kullan.
        "pruned_pt": [
            PROJECT_ROOT / "runs_pruing/finetune/weights/best.pt",
            PROJECT_ROOT / "runs_pruing/yolo_fir_v5s/pruned_final.pt",
        ],
    },
    {
        "key": "faster_rcnn_resnet50",
        "type": "frcnn",
        "imgsz": 800,
        "params_m": 41.31,
        "params_m_pruned": 34.8,
        "normal_pt": [
            PROJECT_ROOT / "runs/faster_rcnn_resnet50_28.04.2026/best.pt",
        ],
        "pruned_pt": [
            PROJECT_ROOT / "runs_pruing/faster_rcnn_resnet50/pruned_final.pt",
        ],
    },
    {
        "key": "faster_rcnn_mobilenet",
        "type": "frcnn",
        "imgsz": 800,
        "params_m": 18.94,
        "params_m_pruned": 14.5,
        "normal_pt": [
            PROJECT_ROOT / "runs/faster_rcnn_mobilenet_scratch_12.05.2026/best.pt",
        ],
        "pruned_pt": [
            PROJECT_ROOT / "runs_pruing/faster_rcnn_mobilenet/pruned_final.pt",
        ],
    },
    {
        "key": "rtdetr_v2_r18vd",
        "type": "rtdetr",
        "imgsz": 480,    # egitim IMAGE_SIZE=480 (640 verirsen guven ~0'a duser)
        "params_m": 20.08,
        "params_m_pruned": 14.3,
        "normal_pt": [
            PROJECT_ROOT / "runs/rtdetr_v2_r18vd_from_scratch_26.05.2026/best.pt",
        ],
        # UYARI: prune_rtdetr_v2_r18vd.py DO_FINETUNE=False -> pruned_final.pt
        # FINETUNE'SUZ, dogrulugu ~0. Anlamli pruned mAP icin finetune sart.
        "pruned_pt": [
            PROJECT_ROOT / "runs_pruing/rtdetr_v2_r18vd/finetune/best.pt",
            PROJECT_ROOT / "runs_pruing/rtdetr_v2_r18vd/pruned_final.pt",
        ],
    },
]


# =====================================================================
# Varyant x Platform matrisi
# =====================================================================
# Kullanicinin istedigi 3 "varyant" her platformda su precision'a denk gelir:
#
#   variant     server (CUDA/CPU)   jetson (TensorRT)   rpi5+hailo (HailoRT)
#   ---------   -----------------   -----------------   ---------------------
#   normal      duz  @ fp32         duz  @ fp16         duz  @ int8 (.hef)
#   pruned      pruned @ fp32       pruned @ fp16       pruned @ int8 (.hef)
#   quantized   pruned @ fp16 (*)   pruned @ int8       pruned @ int8 (==pruned)
#
# (*) Sunucuda gercek INT8 yok (TRT/Hailo gerekir); fp16-GPU "quantize" proxy'sidir.
#     Hailo'da her sey zaten int8 -> "quantized" == "pruned"; ayrica calistirilmaz.
#
# Her giris: (variant, structure, precision, backend_kind)
#   structure    : "normal" | "pruned"  (hangi .pt/.onnx)
#   precision    : "fp32" | "fp16" | "int8"  (raporlama + backend davranisi)
#   backend_kind : "torch" | "trt" | "hailo"
PLATFORM_VARIANTS = {
    # Quantize kategorisi HEM fp16 HEM int8 olarak olculur ("hangi quantize?" sorusu).
    "server": [
        ("normal",     "normal", "fp32", "torch"),
        ("pruned",     "pruned", "fp32", "torch"),
        ("quant_fp16", "pruned", "fp16", "torch"),
        ("quant_int8", "pruned", "int8", "torch"),   # torch'ta int8 yok -> SKIP (not duser)
    ],
    "jetson": [
        ("normal",     "normal", "fp16", "trt"),
        ("pruned",     "pruned", "fp16", "trt"),
        ("quant_fp16", "pruned", "fp16", "trt"),
        ("quant_int8", "pruned", "int8", "trt"),
    ],
    "rpi5_hailo": [
        # backend "rpi": .hef varsa Hailo-8L NPU (int8), yoksa onnxruntime-CPU'ya duser
        # (base/pruned -> fp32 onnx, quant_fp16 -> fp16 onnx, quant_int8 -> int8 onnx).
        # Boylece Hailo derlemesi olmadan da klasor Pi'de calisir.
        ("normal",     "normal", "fp32", "rpi"),
        ("pruned",     "pruned", "fp32", "rpi"),
        ("quant_fp16", "pruned", "fp16", "rpi"),
        ("quant_int8", "pruned", "int8", "rpi"),
    ],
}


# =====================================================================
# Yol cozumleyiciler
# =====================================================================

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def model_dir(key: str) -> Path:
    return MODELS_DIR / key


def canonical_weight(key: str, structure: str, suffix: str = "pt") -> Path:
    # models/<key>/{base,pruned}.{suffix}  (prepare_models.py / convert bunlari uretir).
    return model_dir(key) / f"{STRUCT_LABEL[structure]}.{suffix}"


def onnx_path(key: str, structure: str, precision: str = "fp32") -> Path:
    # models/<key>/base.onnx | base_fp16.onnx | base_int8.onnx (pruned icin de ayni).
    lbl = STRUCT_LABEL[structure]
    if precision in ("fp16", "int8"):
        return model_dir(key) / f"{lbl}_{precision}.onnx"
    return model_dir(key) / f"{lbl}.onnx"


def resolve_source_pt(model: dict, structure: str):
    # Once models/<key>/ icindeki kanonik kopya, yoksa orijinal runs/ adaylari.
    canon = canonical_weight(model["key"], structure, "pt")
    if canon.exists():
        return canon
    cands = model["normal_pt"] if structure == "normal" else model["pruned_pt"]
    return first_existing(cands)


def resolve_device_artifact(model: dict, structure: str, backend_kind: str,
                            precision: str = None):
    # Cihaz backend'leri derlenmis artifact tuketir (models/<key>/ icinde):
    #   trt + yolo        -> <lbl>_<precision>.engine     (precision engine'e gomulu)
    #   trt + frcnn/rtdetr-> <lbl>.onnx                    (fp16/int8 EP flag'i ile)
    #   hailo             -> <lbl>.hef                     (Hailo DFC int8)
    #   rpi / onnx_cpu    -> <lbl>.onnx | <lbl>_fp16.onnx | <lbl>_int8.onnx
    key, d = model["key"], model_dir(model["key"])
    lbl = STRUCT_LABEL[structure]
    if backend_kind == "trt":
        if model["type"] == "yolo":
            return d / f"{lbl}_{precision or 'fp16'}.engine"
        return d / f"{lbl}.onnx"
    if backend_kind == "hailo":
        return d / f"{lbl}.hef"
    if backend_kind in ("rpi", "onnx_cpu"):
        return onnx_path(key, structure, precision or "fp32")
    return canonical_weight(key, structure, "pt")

"""RT-DETR v2 launcher.

Kullanim:
    cd models/rtdetr_v2_r18vd
    python run_training.py

RUN_TYPE'i degistirerek farkli konfigurasyonlar arasinda gec.
"""

import sys
from pathlib import Path

# Sadece bu modulun dizinini path'e ekle.
# PROJECT_ROOT'u EKLEME: yerel "datasets/" klasoru HuggingFace datasets
# paketini golgeler ve HF Trainer 'datasets.Dataset' aradiginda patlar.
sys.path.insert(0, str(Path(__file__).parent))

from train_rtdetr import train


# Hangi konfigurasyonun calisacagini sec.
#   "from_scratch" - HF Hub'tan SADECE config; agirliklar RANDOM init.
#                    Hicbir pretrain yok, epoch 1'den kendi verinle baslar.
#   "finetune"     - HF Hub'tan pretrained agirliklarla baslar (transfer learning).
#   "resume"       - Mevcut checkpoint'ten devam eder (LEGACY, eski dataset).
# Komut satirindan secilebilir: python run_training.py from_scratch
RUN_TYPE = sys.argv[1] if len(sys.argv) > 1 else "finetune"


# restratified_coco: yeniden stratifiye edilmis dataset (train 35190 / val 4470 / test 4480).
# Eski dataset: dataset/2x_augmented_coco_dataset/dataset_augmented
DATA_DIR = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/"
    "dataset/restratified_coco"
)

# Tum yeni egitimler runs_new/ altina kaydedilir.
RUNS_NEW = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs_new"
)

# Yarim kalan run; Excel son satiri epoch 36.81 step 98000, best 0.5375 @ epoch 27 step 72000.
# checkpoint-98000'den devam edilir (Excel/best/last ayni run dizinine append).
EXISTING_CKPT = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/"
    "runs/rtdetr_v2_r18vd_from_scratch_26.05.2026/checkpoints/checkpoint-98000"
)

# Eski 640px thermal-augmented checkpoint (referans icin tutuldu, kullanilmiyor).
OLD_THERMAL_CKPT = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/"
    "models/rtdetr_v2_r18vd/rtdetr_v2_r18vd_thermal_augmented/checkpoint-394400"
)


# --- H200 MIG 1g.35gb (KUCUK SLICE, 35 GB VRAM, full H200'un ~1/7'si) ---
#
# Onceki tur "buyuk batch + buyuk LR" denedi; bottleneck CPU degildi (kullanim
# %11), bottleneck KUCUK MIG slice'in GPU kompute kapasitesi. Dolayisiyla:
#   - IMAGE_SIZE 640 -> 480: ~2.3x daha az piksel -> ~2x throughput
#   - PER_DEVICE_TRAIN_BS 32 -> 16: kucuk SM sayisinda (~26 SM) buyuk batch
#     iyi paralel olmuyor; ufak batch + cok step daha verimli
#   - GRAD_ACCUM 1: degisme
#   - LR oransal duşer (batch 32 -> 16: yarila)
#   - WORKERS 12 -> 6: snapshot'ta 8/12 worker idle idi
#   - eval_strategy "epoch" -> "steps", her ~1.5 epoch (eski: her epoch eval
#     1596 val + slow GPU = ~5-10 min ek yuk per epoch)
#   - torch.compile (DEFAULT_CFG["COMPILE"]=True): ~1.3-2x ek hizlanma


# Random init, epoch 1'den kendi verinle baslar.
FROM_SCRATCH_CFG = {
    "EPOCHS":              300,
    "IMAGE_SIZE":          480,
    "PER_DEVICE_TRAIN_BS": 16,
    "PER_DEVICE_EVAL_BS":  32,
    "GRAD_ACCUM":          1,
    "LR":                  2e-4,        # batch 32 -> 16: 4e-4 yarila dustu
    "WARMUP_STEPS":        1500,
    "WEIGHT_DECAY":        1e-4,
    "PATIENCE":            30,
    "WORKERS":             6,
    "PREFETCH_FACTOR":     2,
    "EVAL_ACCUM_STEPS":    4,
    "EVAL_STEPS":          2000,
    "COMPILE":             False,    # RT-DETR v2 encoder ile uyumsuz; deneme icin True yap
    "PROJECT":             RUNS_NEW,
    "NAME":                "rtdetr_v2_r18vd_from_scratch_restratified_17.06.2026",
    "RESUME_FROM":         None,
    "FROM_SCRATCH":        True,
    "SKIP_CONFIRM":        True,    # 1/0 sormadan cfg'deki FROM_SCRATCH'i kullan
}


# HF Hub'tan pretrained baz model ile fine-tune.
FINETUNE_CFG = {
    "EPOCHS":              150,
    "IMAGE_SIZE":          480,
    "PER_DEVICE_TRAIN_BS": 64,
    "PER_DEVICE_EVAL_BS":  16,    # 32 OOM'd during first eval (Hungarian matcher VRAM spike)
    "GRAD_ACCUM":          1,
    "LR":                  2e-4,    # linear scale: batch 32->64 (x2) -> LR x2
    "WARMUP_STEPS":        500,
    "PATIENCE":            20,
    "WORKERS":             6,
    "PREFETCH_FACTOR":     2,
    "EVAL_ACCUM_STEPS":    4,
    "EVAL_STEPS":          2000,
    "COMPILE":             False,
    "PROJECT":             RUNS_NEW,
    "NAME":                "rtdetr_v2_r18vd_finetune_restratified_01.07.2026",
    "RESUME_FROM":         None,
    "FROM_SCRATCH":        False,
    "SKIP_CONFIRM":        True,
}


# LEGACY: yarim kalan ESKI run'dan (checkpoint-98000) devam.
# DIKKAT: bu config ESKI dataset (2x_augmented) icin; yukaridaki DATA_DIR artik
# restratified_coco'ya isaret ediyor. Resume kullanacaksan DATA_DIR'i eski
# dataset'e geri al, aksi halde farkli veriyle devam etmis olursun.
# Excel son satiri: epoch 36.81 / step 98000. Best: 0.5375 @ epoch 27 / step 72000.
RESUME_CFG = {
    "EPOCHS":              300,         # ilk run'in epoch budget'i ile ayni
    "IMAGE_SIZE":          480,         # ilk run 480px ile egitilmis
    "PER_DEVICE_TRAIN_BS": 16,
    "PER_DEVICE_EVAL_BS":  32,
    "GRAD_ACCUM":          1,
    "LR":                  2e-4,        # ilk run ile ayni LR
    "WARMUP_STEPS":        0,           # resume'da warmup yok
    "WEIGHT_DECAY":        1e-4,
    "PATIENCE":            20,
    "WORKERS":             6,
    "PREFETCH_FACTOR":     2,
    "EVAL_ACCUM_STEPS":    4,
    "EVAL_STEPS":          2000,
    "COMPILE":             False,
    "NAME":                "rtdetr_v2_r18vd_from_scratch_26.05.2026",  # mevcut dir adi (ironik)
    "RESUME_FROM":         EXISTING_CKPT,
    "FROM_SCRATCH":        False,
}


CONFIGS = {
    "from_scratch": FROM_SCRATCH_CFG,
    "finetune":     FINETUNE_CFG,
    "resume":       RESUME_CFG,
}


if __name__ == "__main__":
    if RUN_TYPE not in CONFIGS:
        raise ValueError(f"RUN_TYPE one of {list(CONFIGS.keys())}, got {RUN_TYPE!r}")
    train(data_dir=DATA_DIR, cfg=CONFIGS[RUN_TYPE])

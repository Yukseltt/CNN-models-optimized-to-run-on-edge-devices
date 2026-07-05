"""DETR (classic) launcher.

Kullanim:
    cd models/detr
    python run_training.py

RUN_TYPE'i degistirerek farkli konfigurasyonlar arasinda gec.
"""

import sys
from pathlib import Path

# Sadece bu modulun dizinini path'e ekle.
# PROJECT_ROOT'u EKLEME: yerel "datasets/" klasoru HuggingFace datasets
# paketini golgeler ve HF Trainer 'datasets.Dataset' aradiginda patlar.
sys.path.insert(0, str(Path(__file__).parent))

from train_detr import train


# Hangi konfigurasyonun calisacagini sec.
#   "from_scratch" - HF Hub'tan SADECE config; agirliklar RANDOM init.
#                    Hicbir pretrain yok, epoch 1'den kendi verinle baslar.
#   "finetune"     - HF Hub'tan pretrained agirliklarla baslar (transfer learning).
#   "resume"       - Yarida kalan bir run'dan devam et (RESUME_CFG icine
#                    kendi checkpoint dizinini gir).
RUN_TYPE = sys.argv[1] if len(sys.argv) > 1 else "finetune"

DATA_DIR = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/"
    "dataset/restratified_coco"
)

RUNS_NEW = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs_new"
)

# Resume etmek istersen kendi checkpoint-N dizinini buraya yaz.
# Hic resume gerekmiyorsa bu degisken kullanilmaz (RUN_TYPE != "resume").
EXISTING_CKPT = (
    "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/"
    "runs/detr_finetune_29.05.2026/checkpoints/checkpoint-18000"
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
    "COMPILE":             False,    # DETR (classic) encoder ile uyumsuz; deneme icin True yap
    "NAME":                "detr_from_scratch_29.05.2026",
    "RESUME_FROM":         None,
    "FROM_SCRATCH":        True,
}


# HF Hub'tan pretrained baz model ile fine-tune.
FINETUNE_CFG = {
    "EPOCHS":              150,
    "IMAGE_SIZE":          480,
    "PER_DEVICE_TRAIN_BS": 32,
    "PER_DEVICE_EVAL_BS":  32,
    "GRAD_ACCUM":          1,
    "LR":                  1e-4,    # DETR paper LR — do NOT scale with batch (AdamW normalizes grad)
    "WARMUP_STEPS":        500,
    "PATIENCE":            20,
    "WORKERS":             6,
    "PREFETCH_FACTOR":     2,
    "EVAL_ACCUM_STEPS":    4,
    "EVAL_STEPS":          2000,
    "COMPILE":             False,
    "PRECISION":           "fp32",   # fp16 degenerate boxes crash giou at high grad_norm
    "PROJECT":             RUNS_NEW,
    "NAME":                "detr_finetune_restratified_01.07.2026",
    "RESUME_FROM":         None,
    "FROM_SCRATCH":        False,
    "SKIP_CONFIRM":        True,
}


# Yarim kalan bir run'dan devam et. NAME mevcut run dizinine isaret etmeli,
# RESUME_FROM o run icindeki checkpoint-N dizinini gostermeli.
# Excel/best/last ayni run dizinine append edilir.
RESUME_CFG = {
    "EPOCHS":              150,         # ilk run'in epoch budget'i ile ayni olmali
    "IMAGE_SIZE":          480,
    "PER_DEVICE_TRAIN_BS": 16,
    "PER_DEVICE_EVAL_BS":  32,
    "GRAD_ACCUM":          1,
    "LR":                  1e-4,        # ilk run ile ayni LR olmali
    # _ResumeStateTrainer optimizer'i FRESH (Adam m=0,v=0) basliyor. Warmup=0
    # ile ilk adim ~lr*sign(grad) buyuklugunde TUM parametreleri birden iter
    # (Adam grad buyuklugunu normalize eder) -> converged model saglikli
    # basenden cikar -> sonraki her batch'te NaN gradient -> egitim donar.
    # Warmup, fresh optimizer'in v (2. moment) tahminini kucuk adimlarla
    # kurmasina izin verir; LR 0'dan 1e-4'e tirmanir, ani sicrama olmaz.
    "WARMUP_STEPS":        1000,
    "WEIGHT_DECAY":        1e-4,
    "PATIENCE":            20,
    "WORKERS":             6,
    "PREFETCH_FACTOR":     2,
    "EVAL_ACCUM_STEPS":    4,
    "EVAL_STEPS":          2000,
    "COMPILE":             False,
    # fp16 forward, DETR attention logit'lerinde overflow -> NaN box uretip
    # Hungarian matcher'i compute_loss icinde patlatti (GradScaler sadece
    # backward'i korur, forward crash'ini engelleyemez). fp32 = overflow yok.
    "PRECISION":           "fp32",
    "NAME":                "detr_finetune_29.05.2026",  # mevcut run dizini
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

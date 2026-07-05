#!/usr/bin/env bash
# resume_both_parallel.sh
#
# EfficientNet-B0 detektör + SSD egitimlerini PARALEL (ayni anda) kaldigi
# epoch'tan devam ettirir. Her biri kendi run dizinindeki resume.log'a yazar.
# Resumes the EfficientNet-B0 detector + SSD trainings in PARALLEL from their
# last checkpoints. Each logs to its own run dir's resume.log.
#
# Kullanim / Usage:
#     bash resume_both_parallel.sh
#
# Durdurma / Stop:
#     kill <PID>            (asagida yazdirilan PID'ler)
# Izleme / Monitor:
#     tail -f runs_new/efficientnet_b0_fpn_fasterrcnn_restratified_20.06.2026/resume.log
#     tail -f runs_new/ssdlite_mobilenet_restratified_20.06.2026/resume.log
#
# NOT: Tek MIG dilimi (35GB). VRAM ~14GB(EFFB0)+~8GB(SSD) ~ 22GB < 35GB sigar,
# ama ikisi GPU compute'unu paylasir -> her biri tek basina kosmasindan yavas
# ilerler (toplam is sirali ile benzer). OOM olursa EFFB0 batch'ini dusur.

set -u

ROOT="/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection"
PY="$ROOT/venv/bin/python"

EFFB0_DIR="$ROOT/runs_new/efficientnet_b0_fpn_fasterrcnn_restratified_20.06.2026"
SSD_DIR="$ROOT/runs_new/ssdlite_mobilenet_restratified_20.06.2026"

echo "[resume] EfficientNet-det baslatiliyor (ep 20 -> 150)..."
cd "$ROOT/models/effecientnetB0"
nohup "$PY" resume_training.py efficientnet_imagenet >> "$EFFB0_DIR/resume.log" 2>&1 &
EFFB0_PID=$!
echo "[resume] EFFB0 PID=$EFFB0_PID  log=$EFFB0_DIR/resume.log"

# Iki CUDA context'inin ayni anda init olup birbirini bogmamasi icin kisa ara.
sleep 5

echo "[resume] SSD baslatiliyor (ep 74 -> 300)..."
cd "$ROOT/models/ssd"
nohup "$PY" resume_training.py mobilenet_coco >> "$SSD_DIR/resume.log" 2>&1 &
SSD_PID=$!
echo "[resume] SSD   PID=$SSD_PID  log=$SSD_DIR/resume.log"

echo ""
echo "[resume] Ikisi de arka planda calisiyor."
echo "  Izle:  tail -f $EFFB0_DIR/resume.log"
echo "         tail -f $SSD_DIR/resume.log"
echo "  Durdur: kill $EFFB0_PID $SSD_PID"

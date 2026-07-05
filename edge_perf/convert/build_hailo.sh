#!/usr/bin/env bash
# convert/build_hailo.sh
#
# YOLO ONNX -> Hailo-8L .hef (INT8). x86 DEV makinede, Hailo Dataflow Compiler
# (hailo CLI / hailomz) kurulu ortamda calistir. Ciktiyi RPi5'e kopyala.
#
# Hailo INT8 ZORUNLUDUR (NPU sadece int8 calisir) -> kalibrasyon verisi gerekir
# (~100-500 temsili termal goruntu). frcnn (two-stage) / rtdetr (transformer)
# Hailo-8L'de pratikte derlenmez -> onlar Pi5 CPU onnx ile olculur (main.py halleder).
#
# ON KOSUL:
#   1) edge_perf/convert/export_onnx.py ile YOLO ONNX'leri uret:
#        python edge_perf/convert/export_onnx.py --models yolov8n,yolo_fir_v5s
#      -> weights/yolov8n_normal.onnx, yolov8n_pruned.onnx, ...
#   2) Kalibrasyon goruntuleri: CALIB_DIR icine ~100+ termal jpg koy.
#
# Kullanim:
#   bash edge_perf/convert/build_hailo.sh yolov8n_pruned
#   bash edge_perf/convert/build_hailo.sh yolo_fir_v5s_normal
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="$(cd "$HERE/.." && pwd)/weights"
NAME="${1:?kullanim: build_hailo.sh <key>_<structure>  (orn yolov8n_pruned)}"
ONNX="$WEIGHTS/${NAME}.onnx"
HEF="$WEIGHTS/${NAME}.hef"
HAR="$WEIGHTS/${NAME}.har"
HW_ARCH="hailo8l"
CALIB_DIR="${CALIB_DIR:-$WEIGHTS/calib_imgs}"

[ -f "$ONNX" ] || { echo "ONNX yok: $ONNX  (once export_onnx.py)"; exit 1; }
[ -d "$CALIB_DIR" ] || { echo "Kalibrasyon dizini yok: $CALIB_DIR"; exit 1; }

echo "== Hailo derleme: $NAME (hw=$HW_ARCH) =="

# EN KOLAY YOL: Hailo Model Zoo (hazir YOLO postprocess + NMS on-chip).
if command -v hailomz >/dev/null 2>&1; then
  # Mimari adi: yolov8n / yolov5s ... ckpt olarak ONNX'i ver.
  ARCH_GUESS="yolov8n"; case "$NAME" in *fir*|*v5s*) ARCH_GUESS="yolov5s";; esac
  echo "hailomz compile $ARCH_GUESS ..."
  hailomz compile "$ARCH_GUESS" \
    --ckpt "$ONNX" \
    --calib-path "$CALIB_DIR" \
    --hw-arch "$HW_ARCH" \
    --classes 3
  # hailomz cikti .hef'ini bul ve tasi.
  FOUND="$(ls -t ./*.hef 2>/dev/null | head -1 || true)"
  [ -n "$FOUND" ] && mv "$FOUND" "$HEF" && echo "[OK] -> $HEF"
  exit 0
fi

# ELLE YOL: parser -> optimize (int8 calib) -> compiler.
echo "hailomz yok; elle parser/optimize/compiler..."
hailo parser onnx "$ONNX" --hw-arch "$HW_ARCH" --har "$HAR"
hailo optimize "$HAR" --hw-arch "$HW_ARCH" --calib-set-path "$CALIB_DIR" --output-har-path "$HAR"
hailo compiler "$HAR" --hw-arch "$HW_ARCH" --output-dir "$WEIGHTS"
echo "[OK] -> $HEF (compiler ciktisini kontrol et)"

#!/usr/bin/env python3
# convert/build_tensorrt.py
#
# JETSON ORIN NANO uzerinde calistir. YOLO .pt -> TensorRT .engine.
#   - normal/pruned: FP16 engine
#   - quantized    : INT8 engine (kalibrasyon verisiyle)
#
# frcnn/rtdetr icin AYRI engine build YOK: onnxruntime TensorRT EP, weights/*.onnx'ten
# engine'i ilk run'da JIT-derler ve cache'ler (main.py TrtOnnxBackend halleder).
# Onlar icin once export_onnx.py'yi calistir.
#
# Kullanim (Jetson'da, ultralytics + tensorrt kurulu):
#   sudo jetson_clocks && sudo nvpmodel -m 0
#   python edge_perf/convert/build_tensorrt.py
#   python edge_perf/convert/build_tensorrt.py --models yolov8n --int8-only
#
# INT8 kalibrasyonu icin data.yaml gerekir (ultralytics int8 export). Yoksa
# --no-int8 ile sadece FP16 uret.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Ultralytics INT8 kalibrasyonu icin dataset data.yaml (cihazda gecerli olmali).
DATA_YAML = config.PROJECT_ROOT / "dataset/2x_augmented_yolo_dataset/data.yaml"


def export_engine(pt_path, out_engine, imgsz, int8=False):
    from ultralytics import YOLO
    m = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, device=0)
    if int8:
        kwargs.update(int8=True, data=str(DATA_YAML))
    else:
        kwargs.update(half=True)
    out = m.export(**kwargs)
    Path(out).replace(out_engine)
    print(f"  [OK] -> {out_engine.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--no-int8", action="store_true", help="sadece FP16 engine")
    ap.add_argument("--int8-only", action="store_true", help="sadece INT8 engine")
    args = ap.parse_args()

    sel = (config.MODELS if args.models == "all"
           else [m for m in config.MODELS if m["key"] in args.models.split(",")])
    yolo_models = [m for m in sel if m["type"] == "yolo"]
    if len(yolo_models) < len(sel):
        print("not: frcnn/rtdetr icin export_onnx.py + onnxruntime-TRT EP kullanilir "
              "(bu script sadece YOLO engine uretir).")

    for model in yolo_models:
        # FP16: normal & pruned engine (config: normal->fp16, pruned->fp16)
        if not args.int8_only:
            for structure in ("normal", "pruned"):
                pt = config.resolve_source_pt(model, structure)
                if pt is None:
                    print(f"[YOK] {model['key']}_{structure}"); continue
                eng = config.resolve_device_artifact(model, structure, "trt", "fp16")
                print(f"[{model['key']}_{structure}] FP16 {pt.name} -> {eng.name}")
                export_engine(pt, eng, model["imgsz"], int8=False)

        # INT8: 'quantized' varyant -> pruned yapidan int8 engine.
        if not args.no_int8:
            pt = config.resolve_source_pt(model, "pruned")
            if pt is None:
                print(f"[YOK] {model['key']} pruned (int8 icin)"); continue
            eng = config.resolve_device_artifact(model, "pruned", "trt", "int8")
            print(f"[{model['key']}_quantized] INT8 {pt.name} -> {eng.name}")
            try:
                export_engine(pt, eng, model["imgsz"], int8=True)
            except Exception as e:
                print(f"  [FAIL] int8 export: {e} (data.yaml / kalibrasyon kontrol et)")


if __name__ == "__main__":
    main()

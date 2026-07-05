#!/usr/bin/env python3
# convert/quantize_onnx.py
#
# models/<key>/<lbl>.onnx (fp32) -> _fp16.onnx + _int8.onnx
#   fp16 : onnxconverter_common.float16 (yari hassasiyet, ~2x kucuk)
#   int8 : onnxruntime quantize_dynamic (agirlik int8, kalibrasyonsuz; CPU'da kosar)
#
# Bu .onnx'ler RPi5'te onnxruntime-CPU ile dogrudan calisir (Hailo gerekmez).
# Hailo-8L NPU'da int8 calistirmak istersen .hef sart -> convert/build_hailo.sh.
#
# NOT: int8 icin daha yuksek dogruluk static (QDQ + kalibrasyon) verir; burada
# tasinabilirlik icin dynamic (kalibrasyonsuz) kullanildi. YOLO int8/fp16 onnx
# ultralytics ile her zaman yuklenmeyebilir -> YOLO int8'in asil yolu Hailo .hef.
#
#   python edge_perf/convert/quantize_onnx.py
#   python edge_perf/convert/quantize_onnx.py --models faster_rcnn_resnet50 --no-fp16

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def make_fp16(fp32_path, out_path):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(out_path))


def make_int8(fp32_path, out_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--structures", default="base,pruned")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    sel = (config.MODELS if args.models == "all"
           else [m for m in config.MODELS if m["key"] in args.models.split(",")])
    # disk etiketi (base/pruned) -> ic structure adi
    lbl2struct = {v: k for k, v in config.STRUCT_LABEL.items()}
    structures = [lbl2struct[s] for s in args.structures.split(",") if s in lbl2struct]

    for model in sel:
        for structure in structures:
            fp32 = config.onnx_path(model["key"], structure, "fp32")
            tag = f"{model['key']}/{config.STRUCT_LABEL[structure]}"
            if not fp32.exists():
                print(f"[YOK] {tag}: fp32 onnx yok ({fp32.name}) -> once export_onnx.py")
                continue
            if not args.no_fp16:
                out = config.onnx_path(model["key"], structure, "fp16")
                try:
                    make_fp16(fp32, out); print(f"[OK] {tag} -> {out.name}")
                except Exception as e:
                    print(f"[FAIL] {tag} fp16: {e}")
            if not args.no_int8:
                out = config.onnx_path(model["key"], structure, "int8")
                try:
                    make_int8(fp32, out); print(f"[OK] {tag} -> {out.name}")
                except Exception as e:
                    print(f"[FAIL] {tag} int8: {e}")


if __name__ == "__main__":
    main()

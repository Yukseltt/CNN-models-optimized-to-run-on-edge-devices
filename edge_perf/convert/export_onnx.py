#!/usr/bin/env python3
# convert/export_onnx.py
#
# 5 modelin normal + pruned .pt -> ONNX. Cikti: edge_perf/weights/<key>_<structure>.onnx
#
# Kullanim (sunucuda, torch+CUDA ile):
#   python edge_perf/convert/export_onnx.py                 # hepsi
#   python edge_perf/convert/export_onnx.py --models rtdetr_v2_r18vd --structures pruned
#
# Bu ONNX'ler:
#   - Jetson : onnxruntime TensorRT EP (frcnn/rtdetr) JIT-derler
#   - Pi5    : CPU onnxruntime fallback (frcnn/rtdetr)
#   - Hailo  : YOLO ONNX'i Hailo DFC'ye girdi (build_hailo.sh)
#
# NOT: frcnn (dinamik output) ve rtdetr ONNX export'u kirilgan olabilir; her model
# try/except ile sarili, basarisizsa atlanir ve digerlerine devam eder.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FRCNN_DIR = config.PROJECT_ROOT / "models" / "faster_rcnn"


def export_yolo(pt_path, onnx_path, imgsz):
    from ultralytics import YOLO
    m = YOLO(str(pt_path))
    out = m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    Path(out).replace(onnx_path)
    print(f"  [OK] yolo onnx -> {onnx_path.name}")


def export_frcnn(pt_path, onnx_path, imgsz):
    import torch
    if str(FRCNN_DIR) not in sys.path:
        sys.path.insert(0, str(FRCNN_DIR))
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("model_obj") is not None:
        model = ckpt["model_obj"]
    else:
        from model_loaders import load_faster_rcnn
        model, _names, _cfg = load_faster_rcnn(pt_path, "cpu")
    model.eval()
    dummy = [torch.randn(3, imgsz, imgsz)]
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                      input_names=["images"],
                      output_names=["boxes", "labels", "scores"])
    print(f"  [OK] frcnn onnx -> {onnx_path.name}")


def export_rtdetr(pt_path, onnx_path, imgsz):
    import torch
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("model_obj") is not None:
        model = ckpt["model_obj"]
    else:
        from transformers import AutoConfig, AutoModelForObjectDetection
        names = ckpt.get("class_names", list(config.CLASS_NAMES))
        cfg = AutoConfig.from_pretrained(config.RTDETR_HF_CHECKPOINT)
        cfg.id2label = {i: n for i, n in enumerate(names)}
        cfg.label2id = {n: i for i, n in enumerate(names)}
        cfg.num_labels = len(names)
        model = AutoModelForObjectDetection.from_config(cfg)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=False)
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=17,
                      input_names=["pixel_values"],
                      output_names=["logits", "pred_boxes"],
                      dynamic_axes={"pixel_values": {0: "batch"}})
    print(f"  [OK] rtdetr onnx -> {onnx_path.name}")


EXPORTERS = {"yolo": export_yolo, "frcnn": export_frcnn, "rtdetr": export_rtdetr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--structures", default="normal,pruned")
    args = ap.parse_args()

    sel = (config.MODELS if args.models == "all"
           else [m for m in config.MODELS if m["key"] in args.models.split(",")])
    structures = args.structures.split(",")

    for model in sel:
        config.model_dir(model["key"]).mkdir(parents=True, exist_ok=True)
        for structure in structures:
            pt = config.resolve_source_pt(model, structure)
            tag = f"{model['key']}/{config.STRUCT_LABEL[structure]}"
            if pt is None:
                print(f"[YOK] {tag}: kaynak .pt yok"); continue
            onnx_path = config.canonical_weight(model["key"], structure, "onnx")
            print(f"[{tag}] {pt.name} -> {onnx_path.name}")
            try:
                EXPORTERS[model["type"]](pt, onnx_path, model["imgsz"])
            except Exception as e:
                print(f"  [FAIL] {tag}: {e}")


if __name__ == "__main__":
    main()

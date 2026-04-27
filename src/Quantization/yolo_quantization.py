"""
# ==================== KULLANIM ====================


    q1 = Quantization(
        input_model_path="Quantization/yolov8n.pt",
        output_model_path="Quantization/yolov8n_int8.onnx",
        weight_type="int8", or "int16"
    )
    q1.onnx_ultralytics()

    # Faster R-CNN örneği
    q2 = Quantization(
        input_model_path="Quantization/frcnn_best.pt",
        output_model_path="Quantization/frcnn_int8.onnx",
        weight_type="int8", or "int16"
    )
    q2.onnx_frcnn()
"""

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import RTDETR, YOLO


class Quantization:
    def __init__(self, input_model_path, output_model_path, weight_type):
        self.input_model_path = Path(input_model_path)
        self.output_model_path = Path(output_model_path)
        self.img_size = 640
        self.opset = 19
        self.dynamic = False

        if weight_type == "int8":
            self.weight_type = QuantType.QInt8
        elif weight_type == "int16":
            self.weight_type = QuantType.QInt16

    def onnx_ultralytics(self, simplify=True):
        # opset = ONNX sürümü
        # dynamic = True → input değişken, False → sabit
        # simplify = True → onnxslim ile graph sadeleştirme
        model = YOLO(str(self.input_model_path))
        onnx_path = model.export(
            format="onnx",
            imgsz=self.img_size,
            simplify=simplify,
            opset=self.opset,
            dynamic=self.dynamic,
        )
        onnx_path = Path(onnx_path)

        pre = self._preprocess(onnx_path)
        self._quantize(pre)
        return self.output_model_path


    def onnx_frcnn(self):
        obj = torch.load(
            self.input_model_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(obj, dict):
            raise RuntimeError(
                f"{self.input_model_path} state_dict. Mimari factory gerekir "
                "(ör. fasterrcnn_resnet50_fpn + num_classes)."
            )
        model = obj.eval()
        onnx_path = self.input_model_path.with_suffix(".onnx")
        # 1=batch, 3=kanal, img_size×img_size
        dummy = torch.randn(1, 3, self.img_size, self.img_size)

        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=self.opset,
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes=None,  # input boyutu sabit
        )

        pre = self._preprocess(onnx_path)
        self._quantize(pre)
        return self.output_model_path


    def _preprocess(self, onnx_path):
        # Shape inference + graph optimization. Quantization öncesi zorunlu.
        out = onnx_path.with_name(onnx_path.stem + "_preprocessed.onnx")
        quant_pre_process(
            input_model=str(onnx_path),
            output_model_path=str(out),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,  # TopK/NMS symbolic infer patlıyor
        )
        return out

    def _quantize(self, preprocessed_path):
        quantize_dynamic(
            model_input=str(preprocessed_path),
            model_output=str(self.output_model_path),
            weight_type=self.weight_type,
        )


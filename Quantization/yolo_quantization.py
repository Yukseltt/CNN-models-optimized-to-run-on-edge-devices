"""
==================== KULLANIM ====================

if __name__ == "__main__":
    # YOLO örneği
    q1 = Quantization(
        input_model_path="Quantization/yolov8n.pt",
        output_model_path="Quantization/yolov8n_int8.onnx",
    )
    q1.onnx_ultralytics()

    # Faster R-CNN örneği
    q2 = Quantization(
        input_model_path="Quantization/frcnn_best.pt",
        output_model_path="Quantization/frcnn_int8.onnx",
    )
    q2.onnx_frcnn()
"""

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import RTDETR, YOLO


class Quantization:
    def __init__(self,input_model_path, output_model_path):
        self.input_model_path = Path(input_model_path)
        self.output_model_path = Path(output_model_path)
        self.img_size = 640
        self.opset=19
        self.dynamic = False

    def onnx_ultralytics(self, imgsz, dynamic, simplify=True):
        #opset = onnx  sürümü, 
        # dynamic = False ise model farklı boyutlarda True ise tek boyutta 
        #simplify = True ise onnx optimize et 
        model = YOLO(str(self.input_model_path))
        onnx_path = model.export(format="onnx", imgsz=imgsz, 
            simplify=simplify, opset=self.opset, dynamic=dynamic)

        onnx_path = Path(onnx_path)

        pre = self._pre_process(onnx_path)
        self._quantize(pre)
        return self.output_model_path

    def onnx_frcnn(self):
        obj = torch.load(self.input_model_path,map_location="cpu",
            weights_only=False)
        if isinstance(obj, dict):
            raise RuntimeError(
                f"{self.input_model_path} state_dict. Mimari factory gerekir "
                "(ör. fasterrcnn_resnet50_fpn + num_classes)."
            )
        model = obj.eval()
        onnx_path = self.input_model_path.with_suffix(".onnx")
        dummy = torch.randn(1,3,self.img_size,self.img_size)
        # 1=görüntü, 3=kanal sayısı, img_size=640x640

        torch.onnx.export(model, dummy, onnx_path,
            opset_version=self.opset, input_names=["images"], 
            output_names=["boxes", "labels", "scores"],
            dynamic_axes=None,) #dynamic_axes=input boyutu sabit 
        
        pre = self._pre_process(onnx_path)
        self._quantize(pre)
        return self.output_model_path
    
    def _preprocess(self, onnx_path):
        out = onnx_path.with_name(onnx_path.stem + "preprocessed.onnx")
        quant_pre_process(
            input_model = str(onnx_path),
            output_model = str(out),
            skip_optimization = False,
            skip_onnx_shape = False,
            skip_symbolic_shape_= True,)
        return out
    
    def _quantize(self, preprocessed_path):
        quantize_dynamic(
            model_input = str(preprocessed_path),
            model_output = str(self.output_model_path),
            weight_type = QuantType.QInt8,)
        

from src.Quantization.yolo_quantization import Quantization

q1 = Quantization(
    input_model_path="/home/ugo/Documents/Python/uc_cihaz_obejct_detection/CNN-models-optimized-to-run-on-edge-devices/src/Quantization/Mbest.pt",
    output_model_path="/home/ugo/Documents/Python/uc_cihaz_obejct_detection/CNN-models-optimized-to-run-on-edge-devices/src/Quantization/Mbest_int8.onnx",
)
q1.onnx_ultralytics()
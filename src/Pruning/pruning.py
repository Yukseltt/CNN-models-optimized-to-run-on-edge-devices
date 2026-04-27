"""
Magnitude-based pruning yardımcı sınıfı.

# ==================== KULLANIM ====================

# 1) Ultralytics (.pt) dosyasından doğrudan
    p = Pruning(
        input_model_path="runs/thermal_yolo/weights/best.pt",
        output_model_path="runs/thermal_yolo/weights/best_pruned.pt",
        amount=0.3,                 # %30 ağırlık sıfırla
        method="l1_unstructured",   # veya "ln_structured"
        scope="global",             # veya "local"
    )
    p.prune_ultralytics()

# 2) Faster R-CNN (.pt, tam model veya TorchScript)
    p = Pruning(
        input_model_path="models/faster_cnn4/best.pt",
        output_model_path="models/faster_cnn4/best_pruned.pt",
        amount=0.3,
    )
    p.prune_frcnn()

# 3) Eğitim bittikten sonra bellekteki nn.Module üzerinde
    from src.Pruning.pruning import Pruning
    # model, c = train(cfg)  # yolov11/yolo_train.py
    Pruning.prune_module(
        model.model,                # ultralytics DetectionModel → nn.Module
        amount=0.3,
        method="l1_unstructured",
        scope="global",
    )
    model.save("runs/.../weights/best_pruned.pt")
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import RTDETR, YOLO

model = YOLO("src/Quantization/Mbest.pt")

for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        print(name, "->", layer)

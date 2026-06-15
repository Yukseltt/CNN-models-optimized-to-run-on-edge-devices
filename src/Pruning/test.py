from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import RTDETR, YOLO

model = YOLO("src/Quantization/Mbest.pt")

SKIP_PATTERNS = ("model.0.", "model.23", ".dfl", ".cv2.", ".cv3.")  

prunable = []

for name, layer in model.named_modules():
    if not isinstance(layer, (nn.Conv2d, nn.Linear)):
        continue
    if any(p in name for p in SKIP_PATTERNS):
        continue
    if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
        continue  
    if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels:
        continue  
    prunable.append((name, layer))

for name, layer in prunable:
    print(name, layer.weight.shape)

layer = model.model.model[1].conv   # Conv2d([128, 64, 3, 3])
print(layer.weight.shape)
# L2 norm en düşük %30 output filter sıfırla (dim=0 = out_channels)
prune.ln_structured(layer, name="weight", amount=0.1, n=2, dim=0)



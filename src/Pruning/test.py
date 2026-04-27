import torch
from torch import device, nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from ultralytics import YOLO


model = YOLO("src/Quantization/Mbest.pt").to(device("cuda" if torch.cuda.is_available() else "cpu"))


module = model
print(list(module.modules()))

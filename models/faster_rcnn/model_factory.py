# model_factory.py
#
# Factory for creating Torchvision Faster R-CNN models.
# Torchvision Faster R-CNN modellerini olusturan factory.
#
# Supports two backbones / Iki backbone destekler:
#     - resnet50:  fasterrcnn_resnet50_fpn   (~41M params, baseline)
#     - mobilenet: fasterrcnn_mobilenet_v3_large_fpn  (~19M params, edge)
#
# Both load COCO-pretrained weights, then the classification head is replaced
# to match our num_classes (3 classes + background = 4).
# Iki model de COCO-pretrained agirliklari yukler, sonra classification head
# bizim num_classes degerimizle (3 sinif + background = 4) eslesecek sekilde degistirilir.

from typing import Literal

import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


SupportedBackbone = Literal["resnet50", "mobilenet"]


def create_faster_rcnn(
    backbone:    SupportedBackbone,
    num_classes: int,
    pretrained:  bool = True,
) -> nn.Module:
    # Build a Faster R-CNN model with the requested backbone.
    # Istenen backbone ile bir Faster R-CNN modeli olusturur.
    #
    # Args / Parametreler:
    #     backbone: "resnet50" or "mobilenet".
    #     backbone: "resnet50" veya "mobilenet".
    #
    #     num_classes: total number of output classes including background.
    #     num_classes: background dahil toplam cikti sinif sayisi.
    #         For our dataset: 3 real classes + 1 background = 4.
    #         Bizim dataset icin: 3 gercek sinif + 1 background = 4.
    #
    #     pretrained: load COCO-pretrained weights for the full model.
    #     pretrained: tum model icin COCO-pretrained agirliklari yukle.
    #         The classification head is replaced afterwards either way.
    #         Classification head sonradan zaten degistiriliyor.
    #
    # Returns / Donus:
    #     Torchvision Faster R-CNN module ready for training.
    #     Egitime hazir Torchvision Faster R-CNN modulu.

    if backbone == "resnet50":
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)

    elif backbone == "mobilenet":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            f"Use 'resnet50' or 'mobilenet'."
        )

    # Replace the classification head to match our num_classes.
    # Classification head'i num_classes ile eslesecek sekilde degistir.
    #
    # The pretrained head outputs 91 classes (COCO 2017). We swap it for a
    # fresh predictor sized for our num_classes. The backbone keeps its
    # learned features.
    # Pretrained head 91 sinif (COCO 2017) ciktisi verir. Onu bizim num_classes
    # icin yeni bir predictor ile degistiriyoruz. Backbone ogrendigi feature'lari
    # korur.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> dict:
    # Count trainable and total parameters of a model.
    # Bir modelin egitilebilir ve toplam parametre sayisini sayar.
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":            total,
        "trainable":        trainable,
        "total_millions":     total / 1e6,
        "trainable_millions": trainable / 1e6,
    }
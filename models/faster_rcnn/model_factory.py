# model_factory.py
#
# Factory for creating Torchvision Faster R-CNN models.
# Torchvision Faster R-CNN modellerini olusturan factory.
#
# Supports two backbones / Iki backbone destekler:
#     - resnet50:  fasterrcnn_resnet50_fpn   (~41M params, baseline)
#     - mobilenet: fasterrcnn_mobilenet_v3_large_fpn  (~19M params, edge)
#
# Three pretrained modes / Uc pretrained mode:
#     - "coco":          Full model COCO-pretrained (transfer learning).
#     - "coco":          Tum model COCO-pretrained (transfer learning).
#     - "backbone_only": Only backbone ImageNet-pretrained, FPN/RPN/heads from scratch.
#     - "backbone_only": Sadece backbone ImageNet-pretrained, FPN/RPN/head'ler sifirdan.
#     - None or False:   Everything random.
#     - None veya False: Her sey rastgele.
#
# In all modes, the classification head is replaced to match num_classes.
# Tum modlarda, classification head num_classes ile eslesecek sekilde degistirilir.

from typing import Literal, Union

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
)
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


SupportedBackbone = Literal["resnet50", "mobilenet"]
PretrainedMode    = Union[bool, str, None]


def create_faster_rcnn(
    backbone:    SupportedBackbone,
    num_classes: int,
    pretrained:  PretrainedMode = "coco",
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
    #
    #     pretrained: see module docstring for accepted values.
    #     pretrained: kabul edilen degerler icin modul docstring'ine bakin.
    #         True is treated as "coco" for backward compatibility.
    #         True geriye uyumluluk icin "coco" olarak kabul edilir.

    # Normalize legacy boolean values.
    # Eski boolean degerleri normalize et.
    if pretrained is True:
        pretrained = "coco"
    elif pretrained is False:
        pretrained = None

    if pretrained not in ("coco", "backbone_only", None):
        raise ValueError(
            f"Invalid pretrained mode '{pretrained}'. "
            f"Use 'coco', 'backbone_only', or None."
        )

    if backbone == "resnet50":
        if pretrained == "coco":
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            model = fasterrcnn_resnet50_fpn(weights=weights)
        elif pretrained == "backbone_only":
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
            )
        else:
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )

    elif backbone == "mobilenet":
        if pretrained == "coco":
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        elif pretrained == "backbone_only":
            weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
            )
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=num_classes,
            )

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            f"Use 'resnet50' or 'mobilenet'."
        )

    # Replace the classification head when starting from COCO weights.
    # COCO agirliklarindan baslarken classification head'i degistir.
    #
    # In "backbone_only" and None modes, num_classes was passed at construction,
    # so the head already matches and no replacement is needed.
    # "backbone_only" ve None modlarinda, num_classes constructor'a verildi,
    # head zaten esliyor ve degistirmeye gerek yok.
    if pretrained == "coco":
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
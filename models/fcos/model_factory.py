# model_factory.py
#
# Factory for Torchvision FCOS (Fully Convolutional One-Stage detector).
# Torchvision FCOS (Fully Convolutional One-Stage detector) factory'si.
#
# FCOS is an ANCHOR-FREE single-stage detector (Tian et al., 2019).
# It eliminates anchor boxes entirely: every location on the feature
# map directly predicts a class score, 4 bbox distances to the object's
# borders, and a centerness score. Per-class metrics + IoU loss for boxes
# + focal loss for classification + BCE for centerness.
# FCOS, ANCHOR-FREE tek-asama detector'dur (Tian vd., 2019). Anchor
# box'lari tamamen kaldirir: feature map'teki her konum dogrudan bir
# sinif skoru, objenin kenarlarina 4 bbox mesafesi ve bir centerness
# skoru tahmin eder. Sinif basina metrik + bbox icin IoU loss + sinif
# icin focal loss + centerness icin BCE.
#
# Why add this model: the benchmark has only anchor-based detectors
# (YOLOv8/v11/v26 use anchor-free internally but are not "FCOS-style";
# Faster R-CNN, SSD, RetinaNet all anchor-based). FCOS fills the
# anchor-free CNN gap and provides a clean baseline.
# Bu modeli neden ekledik: benchmark sadece anchor-based detector'lar
# iceriyor (YOLOv8/v11/v26 dahili anchor-free ama "FCOS-style" degil;
# Faster R-CNN, SSD, RetinaNet hepsi anchor-based). FCOS, anchor-free
# CNN bosluugnu doldurur ve temiz bir baseline saglar.
#
# Three pretrained modes / Uc pretrained mode:
#     - "coco":          Full FCOS COCO-pretrained (transfer learning).
#     - "coco":          Tum FCOS COCO-pretrained (transfer learning).
#     - "backbone_only": Only backbone ImageNet-pretrained, head from scratch.
#     - "backbone_only": Sadece backbone ImageNet-pretrained, head sifirdan.
#     - None or False:   Everything random.
#     - None veya False: Her sey rastgele.

from typing import Literal, Union

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
)
from torchvision.models.detection.fcos import FCOSClassificationHead


SupportedBackbone = Literal["resnet50"]
PretrainedMode    = Union[bool, str, None]


def create_fcos(
    backbone:    SupportedBackbone,
    num_classes: int,
    pretrained:  PretrainedMode = "coco",
) -> nn.Module:
    # Build a Torchvision FCOS model.
    # Bir Torchvision FCOS modeli olusturur.
    #
    # Args / Parametreler:
    #     backbone: only "resnet50" supported (torchvision FCOS variant).
    #     backbone: sadece "resnet50" destekleniyor (torchvision FCOS varyanti).
    #
    #     num_classes: total output classes; FCOS uses sigmoid head and
    #         labels are passed as 1-based positive integers (background
    #         is implicit, not a separate class).
    #     num_classes: toplam cikti sinif sayisi; FCOS sigmoid head kullanir
    #         ve label'lar 1-based pozitif tamsayi olarak verilir
    #         (background implicit, ayri bir sinif degil).
    #
    #     pretrained: "coco", "backbone_only", or None.
    #     pretrained: "coco", "backbone_only" veya None.

    if pretrained is True:
        pretrained = "coco"
    elif pretrained is False:
        pretrained = None
    if pretrained not in ("coco", "backbone_only", None):
        raise ValueError(
            f"Invalid pretrained mode '{pretrained}'. "
            f"Use 'coco', 'backbone_only', or None."
        )
    if backbone != "resnet50":
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            f"Only 'resnet50' (torchvision FCOS variant) is available."
        )

    if pretrained == "coco":
        weights = FCOS_ResNet50_FPN_Weights.COCO_V1
        model = fcos_resnet50_fpn(weights=weights)

        # Replace classification head to match num_classes.
        # FCOS head has num_classes channels per anchor location (sigmoid).
        # Classification head'i num_classes ile esitlemek icin degistir.
        # FCOS head'i anchor konumu basina num_classes channel (sigmoid).
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = FCOSClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    elif pretrained == "backbone_only":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    else:
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    return model


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":              total,
        "trainable":          trainable,
        "total_millions":     total / 1e6,
        "trainable_millions": trainable / 1e6,
    }

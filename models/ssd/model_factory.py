# model_factory.py
#
# Factory for the lightest Torchvision SSD variant suitable for edge devices.
# Edge cihazlar icin en hafif Torchvision SSD varyantini olusturan factory.
#
# Backbone / Backbone:
#     - mobilenet_v3_large + SSDLite head  (~3.4M params, 320x320 input)
#     - mobilenet_v3_large + SSDLite head  (~3.4M parametre, 320x320 girdi)
#
# SSDLite uses depthwise-separable convolutions in detection heads and has
# the lightest detection compute among standard SSD variants shipped with
# torchvision. Far smaller than ssd300_vgg16 (~35M params).
# SSDLite, detection head'lerinde depthwise-separable conv kullanir ve
# torchvision'la gelen standart SSD varyantlari arasinda en hafif olanidir.
# ssd300_vgg16'dan (~35M parametre) cok daha kucuktur.
#
# Three pretrained modes / Uc pretrained mode:
#     - "coco":          Full SSDLite COCO-pretrained (transfer learning).
#     - "coco":          Tum SSDLite COCO-pretrained (transfer learning).
#     - "backbone_only": Only backbone ImageNet-pretrained, heads from scratch.
#     - "backbone_only": Sadece backbone ImageNet-pretrained, head'ler sifirdan.
#     - None or False:   Everything random.
#     - None veya False: Her sey rastgele.
#
# In "coco" mode, the classification head is replaced to match num_classes.
# "coco" modunda, classification head num_classes ile eslesecek sekilde
# degistirilir.

from functools import partial
from typing import Literal, Union

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead


SupportedBackbone = Literal["mobilenet"]
PretrainedMode    = Union[bool, str, None]


def create_ssd(
    backbone:    SupportedBackbone,
    num_classes: int,
    pretrained:  PretrainedMode = "coco",
) -> nn.Module:
    # Build the lightest SSD model (SSDLite320 + MobileNetV3-Large).
    # En hafif SSD modelini olusturur (SSDLite320 + MobileNetV3-Large).
    #
    # Args / Parametreler:
    #     backbone: only "mobilenet" is supported (kept for API symmetry with
    #               faster_rcnn factory).
    #     backbone: sadece "mobilenet" destekleniyor (faster_rcnn factory ile
    #               API simetrisi icin tutuluyor).
    #
    #     num_classes: total number of output classes including background.
    #     num_classes: background dahil toplam cikti sinif sayisi.
    #
    #     pretrained: "coco", "backbone_only", or None (True/False legacy ok).
    #     pretrained: "coco", "backbone_only" veya None (True/False eski uyum).

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

    if backbone != "mobilenet":
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            f"Only 'mobilenet' (SSDLite320 + MobileNetV3-Large) is available."
        )

    if pretrained == "coco":
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights)

        # Replace the classification head so num_classes matches our dataset.
        # SSDLite uses a depthwise-separable conv classification head with a
        # specific norm_layer (BatchNorm2d with eps=0.001, momentum=0.03).
        # Classification head'i datasete uyacak sekilde degistir. SSDLite,
        # depthwise-separable conv classification head kullanir ve ozel bir
        # norm_layer ister (BatchNorm2d, eps=0.001, momentum=0.03).
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=norm_layer,
        )

    elif pretrained == "backbone_only":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    else:
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

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

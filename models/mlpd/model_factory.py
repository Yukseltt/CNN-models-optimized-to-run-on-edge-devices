# model_factory.py
#
# Factory for the MLPD-adapted model.
# MLPD-uyarlanmis modelin factory'si.
#
# Important caveat: real MLPD (Kim et al., 2021, "MLPD: Multi-Label
# Pedestrian Detector in Multispectral Domain") is a DUAL-STREAM RGB+T
# detector with cross-modal feature fusion. Our project has THERMAL-ONLY
# data — there is no RGB stream to fuse with. So this is "MLPD-flavored":
# the parts of MLPD we can carry over to thermal-only:
#     - Single-stage detector with sigmoid + focal loss head
#       (RetinaNet head naturally satisfies this)
#     - ResNet50 + FPN backbone (matches original MLPD spec)
#     - Multi-label classification (each anchor independent class scores)
# Onemli uyari: gercek MLPD (Kim vd. 2021) DUAL-STREAM RGB+T detector,
# cross-modal feature fusion ile. Bu projede SADECE TERMAL veri var,
# fuze edilecek RGB stream yok. Bu yuzden bu "MLPD-flavored":
# MLPD'den termal-only'ye tasidiklarimiz:
#     - Tek-asama detector, sigmoid + focal loss head
#       (RetinaNet head bunu dogal olarak saglar)
#     - ResNet50 + FPN backbone (orijinal MLPD spesifikasyonu ile esit)
#     - Multi-label classification (her anchor bagimsiz sinif skoru)
#
# What we do NOT replicate:
#     - Dual-stream architecture (no RGB)
#     - Cross-modal feature fusion modules
#     - KAIST-specific dataset assumptions
# Replike etmediklerimiz:
#     - Dual-stream mimari (RGB yok)
#     - Cross-modal feature fusion modulleri
#     - KAIST'a ozel dataset varsayimlari
#
# Three pretrained modes / Uc pretrained mode:
#     - "coco":          Full RetinaNet COCO-pretrained (transfer learning).
#     - "coco":          Tum RetinaNet COCO-pretrained (transfer learning).
#     - "backbone_only": Only backbone ImageNet-pretrained, head from scratch.
#     - "backbone_only": Sadece backbone ImageNet-pretrained, head sifirdan.
#     - None or False:   Everything random.
#     - None veya False: Her sey rastgele.

from functools import partial
from typing import Literal, Union

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


SupportedBackbone = Literal["resnet50"]
PretrainedMode    = Union[bool, str, None]


def create_mlpd(
    backbone:    SupportedBackbone,
    num_classes: int,
    pretrained:  PretrainedMode = "coco",
) -> nn.Module:
    # Build the MLPD-flavored RetinaNet (ResNet50 + FPN).
    # MLPD-flavored RetinaNet (ResNet50 + FPN) olusturur.
    #
    # Args / Parametreler:
    #     backbone: only "resnet50" supported (matches original MLPD spec).
    #     backbone: sadece "resnet50" destekleniyor (orijinal MLPD ile esit).
    #
    #     num_classes: total output classes; RetinaNet does NOT use a
    #         background label (sigmoid head, multi-label) -> num_classes
    #         here should equal the number of foreground classes.
    #     num_classes: toplam cikti sinif sayisi; RetinaNet background
    #         etiketi KULLANMAZ (sigmoid head, multi-label) -> burada
    #         num_classes foreground sinif sayisina esit olmalidir.
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
            f"Only 'resnet50' (matches original MLPD) is available."
        )

    if pretrained == "coco":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights)

        # Replace the classification head so num_classes matches our dataset.
        # RetinaNet's head has a stack of conv layers + sigmoid output.
        # Classification head'i datasete uyacak sekilde degistir.
        # RetinaNet'in head'i conv stack + sigmoid output icerir.
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        norm_layer = partial(nn.GroupNorm, 32)
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=norm_layer,
        )

    elif pretrained == "backbone_only":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    else:
        model = retinanet_resnet50_fpn_v2(
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

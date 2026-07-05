# detection_model_factory.py
#
# Factory for an EfficientNet-B0 + FPN Faster R-CNN object detector.
# EfficientNet-B0 + FPN Faster R-CNN obje tespit modeli factory'si.
#
# Backbone / Backbone:
#     torchvision efficientnet_b0 (ImageNet1K). FPN, EfficientNet-B0'in
#     features Sequential'inden uc seviye tap'ler:
#         - features[3] -> C3 (stride 8,   40 ch)
#         - features[5] -> C4 (stride 16, 112 ch)
#         - features[7] -> C5 (stride 32, 320 ch)
#     BackboneWithFPN bunlari 256 kanala projeler ve LastLevelMaxPool ile
#     4. seviye ("pool", stride 64) ekler -> toplam 4 FPN seviyesi.
#
# Detection head / Detection head:
#     Standart Torchvision FasterRCNN (RPN + RoIAlign + iki-asamali box head).
#     Loss anahtarlari faster_rcnn ile aynidir:
#         loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
#     boylece train_efficientnet_det.py, train_faster_rcnn.py ile ayni
#     loss-toplama mantigini degismeden kullanir.
#
# Two pretrained modes / Iki pretrained mode:
#     - "imagenet": backbone ImageNet1K-pretrained, FPN/RPN/head'ler sifirdan.
#                   (Torchvision'da COCO-pretrained EfficientNet detektoru yok,
#                    bu yuzden head'ler her zaman sifirdan baslar.)
#     - None / False: her sey rastgele.

from typing import Optional, Union

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


PretrainedMode = Union[bool, str, None]


# EfficientNet-B0 FPN tap noktalari / EfficientNet-B0 FPN tap points.
_RETURN_LAYERS    = {"3": "0", "5": "1", "7": "2"}
_IN_CHANNELS_LIST = [40, 112, 320]
_FPN_OUT_CHANNELS = 256

# EfficientNet-B0'in features Sequential'inde 9 ust-seviye stage var (0..8).
# EfficientNet-B0 has 9 top-level feature stages (features[0..8]).
_NUM_STAGES = 9


def _build_efficientnet_fpn_backbone(
    pretrained_backbone:       bool,
    trainable_backbone_layers: Optional[int],
) -> BackboneWithFPN:
    # EfficientNet-B0 + FPN backbone'u olusturur.
    # Builds the EfficientNet-B0 + FPN backbone.
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    eff = efficientnet_b0(weights=weights)
    features = eff.features

    # Backbone donduruculugu: son `trainable_backbone_layers` stage egitilebilir,
    # gerisi dondurulur. None -> tum backbone egitilebilir (termal domain'e tam
    # adaptasyon icin onerilen, head'ler zaten sifirdan basladigindan).
    # Freeze all but the last `trainable_backbone_layers` stages. None means the
    # whole backbone stays trainable (recommended for the thermal domain shift
    # since the detection heads already start from scratch).
    if trainable_backbone_layers is not None:
        if not (0 <= trainable_backbone_layers <= _NUM_STAGES):
            raise ValueError(
                f"trainable_backbone_layers must be in [0, {_NUM_STAGES}], "
                f"got {trainable_backbone_layers}."
            )
        freeze_before = _NUM_STAGES - trainable_backbone_layers
        for idx, stage in enumerate(features):
            if idx < freeze_before:
                for p in stage.parameters():
                    p.requires_grad_(False)

    # extra_blocks varsayilani LastLevelMaxPool -> "pool" seviyesi eklenir.
    # extra_blocks defaults to LastLevelMaxPool -> adds the "pool" level.
    return BackboneWithFPN(
        features,
        return_layers=_RETURN_LAYERS,
        in_channels_list=_IN_CHANNELS_LIST,
        out_channels=_FPN_OUT_CHANNELS,
    )


def create_efficientnet_fpn_detector(
    num_classes:               int,
    pretrained:                PretrainedMode = "imagenet",
    min_size:                  int = 640,
    max_size:                  int = 640,
    trainable_backbone_layers: Optional[int] = None,
) -> nn.Module:
    # EfficientNet-B0 + FPN tabanli FasterRCNN detektoru olusturur.
    # Builds an EfficientNet-B0 + FPN based FasterRCNN detector.
    #
    # Args / Parametreler:
    #     num_classes: background dahil toplam cikti sinif sayisi.
    #     num_classes: total output classes including background.
    #
    #     pretrained: "imagenet" (backbone ImageNet1K) veya None (rastgele).
    #                 True -> "imagenet", False -> None (geriye uyum).
    #     pretrained: "imagenet" (ImageNet1K backbone) or None (random).
    #                 True -> "imagenet", False -> None (legacy).
    #
    #     min_size / max_size: GeneralizedRCNNTransform yeniden boyutlandirma.
    #         Dataset goruntuleri 640x512 / 640x640 oldugundan varsayilan 640,
    #         native cozunurlugu korur (upscale yapmaz).
    #     min_size / max_size: detection transform resize bounds. Dataset images
    #         are 640x512 / 640x640, so the 640 default keeps native resolution.
    #
    #     trainable_backbone_layers: egitilebilir son stage sayisi (0..9) veya
    #         None (tum backbone egitilebilir).
    #     trainable_backbone_layers: number of trailing trainable stages (0..9)
    #         or None (whole backbone trainable).

    # Eski boolean degerleri normalize et.
    # Normalize legacy boolean values.
    if pretrained is True:
        pretrained = "imagenet"
    elif pretrained is False:
        pretrained = None

    if pretrained not in ("imagenet", None):
        raise ValueError(
            f"Invalid pretrained mode '{pretrained}'. Use 'imagenet' or None."
        )

    backbone = _build_efficientnet_fpn_backbone(
        pretrained_backbone=(pretrained == "imagenet"),
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # 4 FPN seviyesi ("0","1","2","pool", striddeler 8/16/32/64) icin 4 anchor
    # boyutu; her seviyede 3 en-boy orani.
    # 4 anchor sizes for the 4 FPN levels (strides 8/16/32/64); 3 aspect ratios
    # per level.
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "pool"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    # Bir modelin egitilebilir ve toplam parametre sayisini sayar.
    # Count trainable and total parameters of a model.
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":              total,
        "trainable":          trainable,
        "total_millions":     total / 1e6,
        "trainable_millions": trainable / 1e6,
    }

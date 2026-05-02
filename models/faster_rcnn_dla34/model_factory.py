# model_factory.py
#
# Faster R-CNN with DLA-34 backbone + custom FPN.
# DLA-34 backbone + custom FPN ile Faster R-CNN.
#
# DLA-34 backbone uses ImageNet-pretrained weights from timm. The detector head
# is trained from scratch (no COCO Faster R-CNN with DLA-34 in torchvision).
# DLA-34 backbone, timm'den ImageNet-pretrained agirliklari kullanir. Detector
# head sifirdan egitilir (torchvision'da DLA-34'lu COCO Faster R-CNN yok).

from collections import OrderedDict

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


# DLA-34 + FPN backbone wrapper / DLA-34 + FPN backbone sarmalayicisi.

class DLA34FPN(nn.Module):
    # DLA-34 (timm) + Feature Pyramid Network on stages C2..C5.
    # DLA-34 (timm) + C2..C5 stage'leri uzerinde FPN.
    #
    # Output / Cikti: dict with keys "0".."4" (P2..P6), each [B, out_channels, H, W].
    # Output / Cikti: "0".."4" (P2..P6) anahtarli dict, her biri [B, out_channels, H, W].

    def __init__(self, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        # timm import lokal, boylece timm yokken bile dosya import olur.
        import timm

        # out_indices=(2,3,4,5) -> reductions [4, 8, 16, 32] -> channels [64,128,256,512].
        self.body = timm.create_model(
            "dla34",
            features_only=True,
            pretrained=pretrained,
            out_indices=(2, 3, 4, 5),
        )
        in_channels_list = self.body.feature_info.channels()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.body(x)
        x_dict = OrderedDict((str(i), f) for i, f in enumerate(feats))
        return self.fpn(x_dict)


def create_faster_rcnn_dla34(
    num_classes: int,
    pretrained:  bool = True,
) -> nn.Module:
    # Build Faster R-CNN with DLA-34 + FPN backbone.
    # DLA-34 + FPN backbone ile Faster R-CNN olusturur.
    #
    # Args / Parametreler:
    #     num_classes: total number of output classes including background.
    #     num_classes: background dahil toplam cikti sinif sayisi.
    #         For our dataset: 3 real classes + 1 background = 4.
    #         Bizim dataset icin: 3 gercek sinif + 1 background = 4.
    #
    #     pretrained: if True, load ImageNet-pretrained DLA-34 backbone weights.
    #     pretrained: True ise, ImageNet-pretrained DLA-34 backbone yukler.
    backbone = DLA34FPN(pretrained=pretrained, out_channels=256)

    # 5 FPN levels (P2..P6); one anchor size per level, 3 aspect ratios each.
    # 5 FPN seviyesi (P2..P6); seviye basina bir anchor size, her birinde 3 aspect ratio.
    anchor_sizes  = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor    = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    # Count trainable and total parameters of a model.
    # Bir modelin egitilebilir ve toplam parametre sayisini sayar.
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":              total,
        "trainable":          trainable,
        "total_millions":     total / 1e6,
        "trainable_millions": trainable / 1e6,
    }

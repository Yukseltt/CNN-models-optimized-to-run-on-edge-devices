# model_factory.py
#
# U-Net semantic segmentation factory (segmentation_models_pytorch).
# U-Net semantik segmentasyon factory (segmentation_models_pytorch).
#
# Termal goruntuler uzerinde 4-sinif segmentasyon icin kullanilir
# (background / Person / Car / OtherVehicle).

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def create_unet(
    num_classes:     int,
    encoder_name:    str           = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels:     int           = 3,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    # smp.Unet ile U-Net olusturur. encoder_weights="imagenet" ise pretrained
    # encoder agirliklari otomatik indirilir. pretrained_path verilirse, full
    # state_dict oradan yuklenir (resume disinda kullanilir; tipik degil).

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[UNET] Pretrained loaded from {pretrained_path}. "
              f"missing={len(missing)} unexpected={len(unexpected)}")

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

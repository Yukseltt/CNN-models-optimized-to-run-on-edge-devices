# model_factory.py
#
# EfficientNet-B0 image classifier factory.
# EfficientNet-B0 goruntu siniflandirici factory.
#
# Same architecture as the EfficientNet_Cifar100_finetuning.ipynb notebook:
# Swish + SqueezeExcitation + MBConv blocks. Used here to classify thermal
# bbox crops (Person / Car / OtherVehicle) instead of CIFAR-100.
# EfficientNet_Cifar100_finetuning.ipynb notebook'undaki ile ayni mimari:
# Swish + SqueezeExcitation + MBConv block'lari. Burada CIFAR-100 yerine
# termal bbox kirpimlarini siniflandirmak icin kullanilir (Person/Car/OtherVehicle).

import math
import os
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# Pretrained weights URL / Pretrained agirlik URL'i.
PRETRAINED_URL = "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth"


# Building blocks / Yapi taslari.

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes: int, se_planes: int):
        super().__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


class MBConv(nn.Module):
    # Mobile inverted bottleneck conv (MBConv).
    # Mobile inverted bottleneck conv (MBConv).

    def __init__(
        self,
        inplanes:          int,
        planes:            int,
        kernel_size:       int,
        stride:            int,
        expand_rate:       float = 1.0,
        se_rate:           float = 0.25,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes     = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish(),
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inplanes,
                expand_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expand_planes,
                bias=False,
            ),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish(),
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip         = stride == 1
        self.drop_connect_rate = drop_connect_rate

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        # Identity skip / Identity skip.
        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


# Main model / Ana model.

class EfficientNet(nn.Module):
    # Generic EfficientNet (B0..B7 via width/depth coefficients).
    # Genel EfficientNet (width/depth katsayilari ile B0..B7).

    def _setup_repeats(self, num_repeats: int) -> int:
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels: int) -> int:
        num_channels    *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(
        self,
        num_classes:       int   = 1000,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        se_rate:           float = 0.25,
        dropout_rate:      float = 0.2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor           = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides      = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Stem / Stem.
        self.stem = nn.Sequential(
            nn.Conv2d(3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish(),
        )

        # MBConv blocks / MBConv block'lari.
        blocks     = []
        counter    = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):
            num_channels      = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats       = list_num_repeats[idx]
            expand_rate       = expand_rates[idx]
            kernel_size       = kernel_sizes[idx]
            stride            = strides[idx]
            drop_rate         = drop_connect_rate * counter / num_blocks

            blocks.append((
                f"MBConv{expand_rate}_{counter}",
                MBConv(
                    num_channels,
                    next_num_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_rate=expand_rate,
                    se_rate=se_rate,
                    drop_connect_rate=drop_rate,
                ),
            ))
            counter += 1

            for _ in range(1, num_repeats):
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append((
                    f"MBConv{expand_rate}_{counter}",
                    MBConv(
                        next_num_channels,
                        next_num_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        expand_rate=expand_rate,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                ))
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Head / Head.
        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes),
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        return self.head(f)


# Pretrained loading / Pretrained yukleme.

def _download_pretrained(target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path
    print(f"[EFFB0] Downloading pretrained weights to {target_path}")
    urllib.request.urlretrieve(PRETRAINED_URL, str(target_path))
    return target_path


def _load_pretrained_into(model: EfficientNet, weights_path: Path) -> None:
    # Pretrained checkpoint key'leri lokal model key'lerine sirayla maplenir
    # (notebook'taki yontemin aynisi).
    state = torch.load(str(weights_path), map_location="cpu")
    mapping       = {k: v for k, v in zip(state.keys(), model.state_dict().keys())}
    mapped_state  = OrderedDict([(mapping[k], v) for k, v in state.items()])
    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    print(f"[EFFB0] Pretrained loaded. missing={len(missing)} unexpected={len(unexpected)}")


def create_efficientnet_b0(
    num_classes:       int,
    pretrained:        bool          = True,
    pretrained_path:   Optional[str] = None,
    dropout_rate:      float         = 0.2,
    drop_connect_rate: float         = 0.2,
) -> EfficientNet:
    # Build EfficientNet-B0 (1000-class head) and replace head with `num_classes`.
    # EfficientNet-B0'i (1000-sinif head) olusturur; head'i `num_classes` ile degistirir.
    #
    # pretrained=True ise, ImageNet agirliklari yuklenir; dosya yoksa indirilir.

    model = EfficientNet(
        num_classes=1000,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
    )

    if pretrained:
        if pretrained_path is None:
            default_dir = Path(os.environ.get("EFFB0_WEIGHTS_DIR", "/tmp/efficientnet_weights"))
            pretrained_path = str(default_dir / "efficientnet-b0-08094119.pth")
        weights_path = _download_pretrained(Path(pretrained_path))
        _load_pretrained_into(model, weights_path)

    # Yeni siniflandirici head: 1280 -> num_classes.
    in_features         = model.head[6].in_features
    model.head[6]       = nn.Linear(in_features, num_classes)
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

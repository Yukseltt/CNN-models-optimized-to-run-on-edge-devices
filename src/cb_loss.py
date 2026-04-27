# cb_loss.py
#
# Class-Balanced Loss library.
# Class-Balanced Loss kutuphanesi.
#
# Reference / Kaynak:
#     Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
#     https://arxiv.org/abs/1901.05555
#
# Usage (classification) / Kullanim (siniflandirma):
#     criterion = ClassBalancedCrossEntropyLoss(
#         class_counts=[50000, 10000, 3000, 800],
#         beta=0.9999,
#     )
#     loss = criterion(logits, targets)
#
# Usage (detection) / Kullanim (object detection):
#     cls_loss_fn = ClassBalancedDetectionClsLoss(
#         class_counts=[50000, 10000, 3000, 800],
#         beta=0.9999,
#         loss_type="focal",
#     )
#     loss = cls_loss_fn(pred_logits, assigned_labels)
#
# Beta guide / Beta kilavuzu:
#     Higher beta means larger weight gap between classes.
#     Yuksek beta, siniflar arasi agirlik farkini artirir.
#     Typical values: 0.9, 0.99, 0.999, 0.9999.
#     Tipik degerler: 0.9, 0.99, 0.999, 0.9999.
#
# Gamma guide (focal only) / Gamma kilavuzu (sadece focal):
#     Controls focus on hard examples.
#     Zor orneklere odaklanma gucunu belirler.
#     0.0 disables focal term, 2.0 is the standard value.
#     0.0 focal terimini kapatir, 2.0 standart degerdir.

import json as _json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# Auto-compute class counts from a COCO annotation file.
# COCO annotation dosyasindan sinif sayilarini otomatik hesaplar.

def compute_class_counts_from_coco(
    annotation_path: str,
) -> tuple[list[str], list[int]]:
    # Read COCO annotation JSON and return class names and counts.
    # COCO annotation JSON'unu okur, sinif isimleri ve sayilarini dondurur.
    #
    # Returns (class_names, class_counts) sorted by category id.
    # Category id sirasina gore (class_names, class_counts) dondurur.
    with open(annotation_path, "r") as f:
        coco = _json.load(f)

    sorted_cats = sorted(coco["categories"], key=lambda x: x["id"])
    class_names = [cat["name"] for cat in sorted_cats]

    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    class_counts = [cat_counts.get(cat["id"], 0) for cat in sorted_cats]

    return class_names, class_counts


# Core weight computation.
# Cekirdek agirlik hesaplamasi.

def compute_cb_weights(
    class_counts: list[int],
    beta: float = 0.9999,
    normalize: bool = True,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    # Compute Class-Balanced weights from per-class sample counts.
    # Sinif basina ornek sayilarindan Class-Balanced agirliklarini hesaplar.
    #
    # Formula / Formul:
    #     E(n) = (1 - beta^n) / (1 - beta)
    #     w_c  = 1 / E(n_c)
    #
    # If normalize is True, weights sum to num_classes.
    # normalize True ise agirliklarin toplami sinif sayisina esitlenir.
    #
    # Returns a tensor of shape (num_classes,).
    # Shape (num_classes,) olan bir tensor dondurur.
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got: {beta}")
    if any(n <= 0 for n in class_counts):
        raise ValueError("All class_counts values must be positive integers.")

    counts = np.array(class_counts, dtype=np.float64)
    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num

    if normalize:
        weights = weights / weights.sum() * len(class_counts)

    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# Classification loss wrappers.
# Siniflandirma loss sarmalayicilari.

class ClassBalancedCrossEntropyLoss(nn.Module):
    # Class-Balanced weighted Cross Entropy Loss for classification.
    # Siniflandirma icin Class-Balanced agirlikli Cross Entropy Loss.
    #
    # Expects logits of shape (B, C) and targets of shape (B,).
    # (B, C) shape'li logits ve (B,) shape'li targets bekler.

    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        weights = compute_cb_weights(class_counts, beta=beta, device=device)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C) raw model outputs / ham model ciktilari
        # targets: (B,) class indices / sinif indeksleri
        return self.loss_fn(logits, targets)


class ClassBalancedFocalLoss(nn.Module):
    # Class-Balanced weighted Focal Loss for classification.
    # Siniflandirma icin Class-Balanced agirlikli Focal Loss.
    #
    # Combines class imbalance handling and hard-example focusing.
    # Sinif dengesizligi ve zor ornek odaklanmasini birlestirir.
    #
    # Formula / Formul:
    #     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got: {gamma}")
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "weights",
            compute_cb_weights(class_counts, beta=beta, device=device)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C) raw model outputs / ham model ciktilari
        # targets: (B,) class indices / sinif indeksleri
        probs = torch.softmax(logits, dim=1)
        p_t = probs[torch.arange(len(targets), device=logits.device), targets]
        alpha_t = self.weights[targets]
        loss = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Detection loss, model-agnostic classification component.
# Detection icin model-bagimsiz siniflandirma loss bileseni.

class ClassBalancedDetectionClsLoss(nn.Module):
    # Class-Balanced classification loss for object detection.
    # Object detection icin Class-Balanced siniflandirma loss'u.
    #
    # Works with anchor-based, anchor-free and two-stage detectors.
    # Anchor-based, anchor-free ve iki asamali detektorlerle calisir.
    #
    # This class computes only the cls loss; box and obj losses are unchanged.
    # Bu sinif sadece cls loss hesaplar; box ve obj loss degismez.
    #
    # Parameters / Parametreler:
    #     class_counts: per foreground class sample counts (background excluded).
    #     class_counts: foreground sinif basina ornek sayilari (background haric).
    #
    #     loss_type: "focal", "ce" or "bce".
    #     loss_type: "focal", "ce" veya "bce".
    #
    #     background_idx: index of the background class, or None if absent.
    #     background_idx: background sinif indeksi, yoksa None.
    #         YOLO, NanoDet use None; Faster R-CNN uses 0.
    #         YOLO, NanoDet None kullanir; Faster R-CNN 0 kullanir.

    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = "focal",
        background_idx: Optional[int] = None,
        reduction: str = "mean",
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        if loss_type not in ("focal", "ce", "bce"):
            raise ValueError(f"loss_type must be 'focal', 'ce' or 'bce', got: {loss_type}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got: {gamma}")

        self.loss_type = loss_type
        self.gamma = gamma
        self.reduction = reduction
        self.background_idx = background_idx

        fg_weights = compute_cb_weights(class_counts, beta=beta, normalize=True)

        # If background exists, insert weight 1.0 at the background index.
        # Background varsa agirlik vektorune background indeksinde 1.0 ekle.
        if background_idx is not None:
            num_classes = len(class_counts) + 1
            weights = torch.ones(num_classes, dtype=torch.float32)
            fg_indices = [i for i in range(num_classes) if i != background_idx]
            weights[fg_indices] = fg_weights
        else:
            weights = fg_weights

        self.register_buffer("weights", weights.to(device) if device else weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # logits: (N, C) or (N, C+1) raw model outputs.
        # logits: (N, C) veya (N, C+1) ham model ciktilari.
        #
        # targets: (N,) class index per anchor or ROI.
        # targets: (N,) her anchor veya ROI icin sinif indeksi.
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        # Drop background samples if background_idx is set.
        # background_idx verilmisse background orneklerini cikar.
        if self.background_idx is not None:
            fg_mask = targets != self.background_idx
            if fg_mask.sum() == 0:
                # Keep gradient connection / gradient baglantisini koru
                return logits.sum() * 0.0
            logits  = logits[fg_mask]
            targets = targets[fg_mask]

        if self.loss_type == "ce":
            return F.cross_entropy(
                logits, targets,
                weight=self.weights,
                reduction=self.reduction,
            )

        elif self.loss_type == "focal":
            probs = torch.softmax(logits, dim=1)
            p_t = probs[torch.arange(len(targets), device=logits.device), targets]
            alpha_t = self.weights[targets]
            loss = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        else:
            # bce branch for multi-label scenarios.
            # Multi-label senaryolar icin bce dali.
            num_classes = logits.size(1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            alpha_t = self.weights.unsqueeze(0).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(
                logits, one_hot,
                weight=alpha_t,
                reduction="none",
            ).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
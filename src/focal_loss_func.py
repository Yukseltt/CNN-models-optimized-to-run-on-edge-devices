# focal_loss_func.py
#
# Focal Loss for binary classification (BCE-based).
# Binary classification icin Focal Loss (BCE tabanli).
#
# Reference / Kaynak:
#     Lin et al. (2017) "Focal Loss for Dense Object Detection"
#     https://arxiv.org/abs/1708.02002
#
# Formula / Formul:
#     FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
#
# Where p_t is the model's probability for the true class.
# p_t modelin gercek sinifa verdigi olasilik.
#
# Easy examples (p_t close to 1) get reduced loss.
# Kolay ornekler (p_t 1'e yakin) daha az loss alir.
#
# Hard examples (p_t close to 0.5) keep their full loss.
# Zor ornekler (p_t 0.5'e yakin) tam loss'u alir.

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    # Compute Focal Loss for binary classification using logits.
    # Logits kullanarak binary classification icin Focal Loss hesaplar.
    #
    # Args / Parametreler:
    #     logits: raw model outputs before sigmoid, any shape.
    #     logits: sigmoid'den onceki ham model ciktilari, herhangi bir shape.
    #
    #     targets: ground-truth labels, same shape as logits, values in {0, 1}.
    #     targets: gercek etiketler, logits ile ayni shape, degerler {0, 1}.
    #
    #     alpha: balancing factor between positive and negative examples.
    #     alpha: pozitif ve negatif ornekler arasi denge faktoru.
    #         Standard value is 0.25.
    #         Standart degeri 0.25.
    #
    #     gamma: focusing parameter, controls how much to down-weight easy examples.
    #     gamma: odaklanma parametresi, kolay orneklerin agirligini ne kadar dusurecegini belirler.
    #         0.0 disables focal term, 2.0 is the standard value.
    #         0.0 focal terimini kapatir, 2.0 standart degerdir.
    #
    #     reduction: "none", "mean", or "sum".
    #     reduction: "none", "mean" veya "sum".
    #
    # Returns / Donus:
    #     Loss tensor with the same shape as logits if reduction is "none".
    #     reduction "none" ise logits ile ayni shape'de loss tensor.

    # Step 1: Standard BCE per element (no reduction yet).
    # Adim 1: Element basina standart BCE (henuz reduction yok).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Step 2: Compute p_t = probability of the true class.
    # Adim 2: p_t hesapla = gercek sinifin olasiligi.
    #   For positive examples (target=1): p_t = sigmoid(logits)
    #   For negative examples (target=0): p_t = 1 - sigmoid(logits)
    #   Pozitif ornekler (target=1) icin: p_t = sigmoid(logits)
    #   Negatif ornekler (target=0) icin: p_t = 1 - sigmoid(logits)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)

    # Step 3: Focal modulating factor (1 - p_t)^gamma.
    # Adim 3: Focal modulating faktoru (1 - p_t)^gamma.
    #   Easy examples (p_t high) get a small factor, loss is reduced.
    #   Hard examples (p_t low) get a large factor, loss stays high.
    #   Kolay ornekler (p_t yuksek) kucuk faktor alir, loss azalir.
    #   Zor ornekler (p_t dusuk) buyuk faktor alir, loss yuksek kalir.
    focal_factor = (1.0 - p_t) ** gamma

    # Step 4: Alpha balancing.
    # Adim 4: Alpha dengelemesi.
    #   alpha for positives, (1 - alpha) for negatives.
    #   Pozitifler icin alpha, negatifler icin (1 - alpha).
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)

    # Step 5: Combine all parts.
    # Adim 5: Tum parcalari birlestir.
    loss = alpha_factor * focal_factor * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
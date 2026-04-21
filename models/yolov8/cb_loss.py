"""
cb_loss.py — Class-Balanced Loss Kütüphanesi
=============================================
Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
https://arxiv.org/abs/1901.05555

KULLANIM — Temel (classification):
    from cb_loss import ClassBalancedCrossEntropyLoss, ClassBalancedFocalLoss

    criterion = ClassBalancedCrossEntropyLoss(
        class_counts = [50000, 10000, 3000, 800],
        beta         = 0.9999,
    )
    loss = criterion(logits, targets)   # logits: (B,C)  targets: (B,)

KULLANIM — Object detection (anchor/pred bazlı):
    from cb_loss import ClassBalancedDetectionClsLoss

    cls_loss_fn = ClassBalancedDetectionClsLoss(
        class_counts    = [50000, 10000, 3000, 800],
        beta            = 0.9999,
        background_idx  = None,   # arka plan sınıf indeksi (varsa)
        loss_type       = "focal", # "ce" | "focal" | "bce"
    )

    # Anchor-based / tek aşamalı (YOLO, NanoDet, RetinaNet):
    loss = cls_loss_fn(pred_logits, assigned_labels)
    # pred_logits    : (N, C)  — tüm pozitif anchor tahminleri
    # assigned_labels: (N,)    — her anchor'un atandığı sınıf

    # İki aşamalı (Faster R-CNN ROI head):
    loss = cls_loss_fn(roi_logits, roi_labels)
    # roi_logits : (R, C+1)  — ROI classifier çıktıları
    # roi_labels : (R,)      — her ROI'nin etiket indeksi

BETA KILAVUZU
-------------
beta, veri örtüşme derecesini temsil eder (0 < β < 1).

    0.9    : Ağırlıklar birbirine çok yaklaşır. Sınıflar arası fark
             neredeyse kaybolur. Çok fazla augmentation uygulanmış
             veri setlerinde kullanılır.

    0.99   : Orta düzey örtüşme. Dengeli bir başlangıç noktasıdır.

    0.999  : Düşük-orta örtüşme. Çoğu gerçek dünya veri setinde
             iyi bir başlangıç noktasıdır.

    0.9999 : Düşük örtüşme. Sınıflar arası ağırlık farkı belirgindir.
             Ham ve çeşitli veri setleri için önerilir. (varsayılan)

    Kural: Augmentation arttıkça beta'yı düşür,
           veri çeşitliliği arttıkça beta'yı yükselt.

GAMMA KILAVUZU (focal loss için)
----------------------------------
gamma, focal loss'un "zor örnek odaklanma" gücünü belirler (γ ≥ 0).

    0.0 : Focal terimi devre dışı. CB ağırlıklı CE'ye eşdeğer.
    1.0 : Hafif odaklanma.
    2.0 : Standart değer (RetinaNet varsayılanı). Önerilen başlangıç.
    5.0 : Güçlü odaklanma. Çok zor ya da gürültülü veri setlerinde
          modeli kararsızlaştırabilir, dikkatli kullanın.
"""

import json as _json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────
# COCO'dan Otomatik Sınıf Sayısı Hesaplama
# ──────────────────────────────────────────────────────────────

def compute_class_counts_from_coco(
    annotation_path: str,
) -> tuple[list[str], list[int]]:
    """
    COCO annotation JSON dosyasından sınıf isimlerini ve
    annotation sayılarını otomatik hesaplar.

    Parametreler
    ------------
    annotation_path : str
        COCO formatında annotation JSON dosya yolu.
        Örnek: "dataset/train/_annotations.coco.json"

    Döndürür
    --------
    (class_names, class_counts) : tuple[list[str], list[int]]
        class_names  : Sınıf isimleri (category id sırasına göre)
        class_counts : Her sınıfa ait annotation sayısı

    Örnek
    -----
        names, counts = compute_class_counts_from_coco("train/_annotations.coco.json")
        # names  = ["Person", "Car", "OtherVehicle"]
        # counts = [335428, 577119, 36186]

        criterion = ClassBalancedFocalLoss(class_counts=counts, beta=0.9999)
    """
    with open(annotation_path, "r") as f:
        coco = _json.load(f)

    sorted_cats = sorted(coco["categories"], key=lambda x: x["id"])
    class_names = [cat["name"] for cat in sorted_cats]

    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    class_counts = [cat_counts.get(cat["id"], 0) for cat in sorted_cats]

    return class_names, class_counts


# ──────────────────────────────────────────────────────────────
# Çekirdek: Ağırlık Hesaplama
# ──────────────────────────────────────────────────────────────

def compute_cb_weights(
    class_counts: list[int],
    beta: float = 0.9999,
    normalize: bool = True,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Class-Balanced ağırlıklarını hesaplar.

    Formül:
        E(n) = (1 - β^n) / (1 - β)   ← etkin örnek sayısı
        w_c  = 1 / E(n_c)             ← sınıf ağırlığı

    Parametreler
    ------------
    class_counts : list[int]
        Her sınıfa ait eğitim örneği sayıları.
        Sıra, modelin çıktı sınıf sırasıyla eşleşmelidir.
        Örnek: [50000, 10000, 3000, 800]

    beta : float, varsayılan 0.9999
        Etkin örnek sayısı hiperparametresi. (0 < β < 1)
        Yüksek değer → sınıflar arası ağırlık farkı artar.
        Düşük değer  → ağırlıklar birbirine yaklaşır.
        Detaylı kılavuz için modül docstring'ine bakın.

    normalize : bool, varsayılan True
        True  → Ağırlıkların toplamı sınıf sayısına eşitlenir.
                 Loss değeri sınıf sayısından bağımsız ölçekte kalır.
        False → Ham ters-etkin-sayı değerleri döndürülür.

    device : torch.device | str | None, varsayılan None
        Tensörün yerleştirileceği cihaz. None ise CPU kullanılır.
        Örnek: "cuda", "cuda:0", torch.device("cuda")

    Döndürür
    --------
    torch.Tensor, shape=(num_classes,)
    """
    if not (0 < beta < 1):
        raise ValueError(f"beta (0, 1) aralığında olmalı, alınan: {beta}")
    if any(n <= 0 for n in class_counts):
        raise ValueError("Tüm class_counts değerleri pozitif tam sayı olmalı.")

    counts = np.array(class_counts, dtype=np.float64)
    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num

    if normalize:
        weights = weights / weights.sum() * len(class_counts)

    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ──────────────────────────────────────────────────────────────
# Loss 1: Classification (standart kullanım)
# ──────────────────────────────────────────────────────────────

class ClassBalancedCrossEntropyLoss(nn.Module):
    """
    Class-Balanced ağırlıklı Cross Entropy Loss.

    Standart sınıflandırma görevleri için. Giriş olarak
    (B, C) logits ve (B,) targets bekler.

    Parametreler
    ------------
    class_counts : list[int]
        Her sınıfa ait eğitim örneği sayıları.

    beta : float, varsayılan 0.9999
        Etkin örnek sayısı hiperparametresi. (0 < β < 1)
        Artırmak → az örnekli sınıflara daha güçlü ağırlık.
        Azaltmak → sınıflar arası fark yumuşar.

    label_smoothing : float, varsayılan 0.0
        Etiket yumuşatma katsayısı. [0.0, 1.0)
        0.0  → yumuşatma yok.
        0.1  → hedef olasılığın %10'u diğer sınıflara dağıtılır.
               Aşırı öğrenmeyi azaltır, genellemeyi artırabilir.

    reduction : str, varsayılan "mean"
        "mean" → batch ortalaması. (önerilen)
        "sum"  → batch toplamı.
        "none" → her örnek için ayrı değer.

    device : torch.device | str | None, varsayılan None

    Örnek
    -----
        criterion = ClassBalancedCrossEntropyLoss(
            class_counts    = [50000, 10000, 3000, 800],
            beta            = 0.9999,
            label_smoothing = 0.1,
        )
        loss = criterion(logits, targets)  # logits:(B,C)  targets:(B,)
    """

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
        """
        logits  : (B, C) — ham model çıktıları
        targets : (B,)   — sınıf indeksleri (long)
        """
        return self.loss_fn(logits, targets)


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced ağırlıklı Focal Loss.

    Sınıf dengesizliği (CB ağırlığı) ve zor örnek odaklanması
    (focal terimi) mekanizmalarını birleştirir.

    Formül:
        FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Parametreler
    ------------
    class_counts : list[int]
        Her sınıfa ait eğitim örneği sayıları.

    beta : float, varsayılan 0.9999
        Etkin örnek sayısı hiperparametresi. (0 < β < 1)

    gamma : float, varsayılan 2.0
        Focal loss odaklanma parametresi. (γ ≥ 0)
        0.0 → Focal terimi devre dışı (CB CE'ye eşdeğer).
        2.0 → Standart RetinaNet değeri. Önerilen başlangıç.
        5.0 → Güçlü odaklanma. Dikkatli kullanın.

    reduction : str, varsayılan "mean"
        "mean" | "sum" | "none"

    device : torch.device | str | None, varsayılan None

    Örnek
    -----
        criterion = ClassBalancedFocalLoss(
            class_counts = [50000, 10000, 3000, 800],
            beta         = 0.9999,
            gamma        = 2.0,
        )
        loss = criterion(logits, targets)  # logits:(B,C)  targets:(B,)
    """

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
            raise ValueError(f"gamma >= 0 olmalı, alınan: {gamma}")
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "weights",
            compute_cb_weights(class_counts, beta=beta, device=device)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C) — ham model çıktıları
        targets : (B,)   — sınıf indeksleri (long)
        """
        probs = torch.softmax(logits, dim=1)
        p_t = probs[torch.arange(len(targets), device=logits.device), targets]
        alpha_t = self.weights[targets]
        loss = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────
# Loss 2: Detection — model-agnostic cls loss
# ──────────────────────────────────────────────────────────────

class ClassBalancedDetectionClsLoss(nn.Module):
    """
    Object Detection için Class-Balanced sınıf (cls) loss.

    Model mimarisinden bağımsız çalışır. Kullanım:
      - Anchor-based tek aşamalı: YOLO, RetinaNet, NanoDet
      - Anchor-free tek aşamalı : FCOS, CenterNet, TOOD
      - İki aşamalı ROI head    : Faster R-CNN, Mask R-CNN

    Detection'da loss pipeline'ı üç parçadan oluşur:
        total_loss = λ_box · L_box + λ_cls · L_cls + λ_obj · L_obj

    Bu sınıf yalnızca L_cls hesaplar. L_box ve L_obj değişmez.
    CB ağırlıkları background dahil tüm sınıflara uygulanır.
    Background'u dışarıda tutmak için background_idx kullanın.

    Parametreler
    ------------
    class_counts : list[int]
        Arka plan HARİÇ, her ön plan sınıfına ait örnek sayıları.
        Örnek: [50000, 10000, 3000, 800]

    beta : float, varsayılan 0.9999
        Etkin örnek sayısı hiperparametresi. (0 < β < 1)
        Artırmak → az örnekli sınıflara daha güçlü ağırlık.
        Azaltmak → sınıflar arası fark yumuşar.

    gamma : float, varsayılan 2.0
        Focal loss odaklanma parametresi. loss_type="focal" için geçerli.
        0.0 → Focal terimi devre dışı.
        2.0 → Standart RetinaNet değeri. Önerilen.

    loss_type : str, varsayılan "focal"
        "focal" → CB ağırlıklı Focal Loss. (detection için önerilen)
                  Hem sınıf dengesizliğini hem zor örnekleri ele alır.
        "ce"    → CB ağırlıklı Cross Entropy. Daha basit, iyi baseline.
        "bce"   → CB ağırlıklı Binary Cross Entropy.
                  Çok etiketli (multi-label) senaryolar için.
                  logits shape: (N, C) — sigmoid uygulanır.

    background_idx : int | None, varsayılan None
        Background sınıfının indeksi. Verilirse:
          - class_counts listesine background eklenmez.
          - CB ağırlık vektörüne background için w=1.0 eklenir.
          - Background örnekleri loss hesaplamasından çıkarılır.
        None → background yoktur ya da dışarıda tutulmuştur.

        Kullanım örnekleri:
          YOLO         : background_idx=None  (objectness ayrı ele alınır)
          Faster R-CNN : background_idx=0     (sınıf 0 = background)
          NanoDet      : background_idx=None  (anchor-free, bg yok)

    reduction : str, varsayılan "mean"
        "mean" → pozitif anchor ortalaması. (önerilen)
        "sum"  → toplam. Batch büyüklüğüne göre normalize gerekebilir.
        "none" → her anchor için ayrı değer.

    device : torch.device | str | None, varsayılan None

    Örnek — YOLO / NanoDet (background yok):
    -----------------------------------------
        cls_loss_fn = ClassBalancedDetectionClsLoss(
            class_counts   = [50000, 10000, 3000, 800],
            beta           = 0.9999,
            gamma          = 2.0,
            loss_type      = "focal",
            background_idx = None,
        )

        # Eğitim döngüsünde:
        # pos_logits : (N_pos, C)  — sadece pozitif anchor logits
        # pos_labels : (N_pos,)    — her anchor'un gt sınıfı
        cls_loss   = cls_loss_fn(pos_logits, pos_labels)
        total_loss = box_loss + cls_loss + obj_loss

    Örnek — Faster R-CNN (background sınıf 0):
    --------------------------------------------
        cls_loss_fn = ClassBalancedDetectionClsLoss(
            class_counts   = [50000, 10000, 3000, 800],
            beta           = 0.9999,
            loss_type      = "ce",
            background_idx = 0,
        )

        # ROI head içinde:
        # roi_logits : (R, C+1)   — background dahil C+1 sınıf
        # roi_labels : (R,)       — 0=background, 1..C=ön plan
        cls_loss = cls_loss_fn(roi_logits, roi_labels)
    """

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
            raise ValueError(f"loss_type 'focal', 'ce' veya 'bce' olmalı, alınan: {loss_type}")
        if gamma < 0:
            raise ValueError(f"gamma >= 0 olmalı, alınan: {gamma}")

        self.loss_type = loss_type
        self.gamma = gamma
        self.reduction = reduction
        self.background_idx = background_idx

        fg_weights = compute_cb_weights(class_counts, beta=beta, normalize=True)

        # Background varsa ağırlık vektörüne ekle (w=1.0)
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
        """
        Parametreler
        ------------
        logits  : torch.Tensor, shape=(N, C) veya (N, C+1)
            Ham model çıktıları. bce için (N, C) — sigmoid uygulanır.

        targets : torch.Tensor, shape=(N,)
            Her anchor/ROI için sınıf indeksi (long dtype).
            background_idx verilmişse o indeksteki örnekler
            loss hesaplamasından otomatik çıkarılır.

        Döndürür
        --------
        torch.Tensor: Skaler loss (reduction="mean" ise).
        """
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        # Background örneklerini çıkar
        if self.background_idx is not None:
            fg_mask = targets != self.background_idx
            if fg_mask.sum() == 0:
                return logits.sum() * 0.0  # gradient bağlantısını koru
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

        else:  # bce
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

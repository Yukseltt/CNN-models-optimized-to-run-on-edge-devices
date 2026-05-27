# eval_metrics.py
#
# Custom precision / recall / F1 evaluation in YOLO style.
# YOLO tarzi custom precision / recall / F1 degerlendirmesi.
#
# Algorithm / Algoritma:
#     1. Sweep multiple confidence thresholds.
#     1. Birden cok confidence threshold dene.
#     2. For each threshold, count TP / FP / FN per class at IoU >= 0.5.
#     2. Her threshold icin, IoU >= 0.5'te per class TP / FP / FN say.
#     3. Pick the threshold that maximizes overall F1.
#     3. Genel F1'i maksimize eden threshold'u sec.
#     4. Return P, R, F1 at that threshold (overall and per-class).
#     4. O threshold'da P, R, F1 dondur (overall ve per-class).
#
# Notes / Notlar:
#     Predictions and targets are kept as numpy arrays.
#     Tahminler ve hedefler numpy array olarak tutulur.
#
#     Boxes are in [x_min, y_min, x_max, y_max] pixel format.
#     Bbox'lar [x_min, y_min, x_max, y_max] pixel formatindadir.
#
#     SSD labels are 1-based (background=0 reserved), same as Faster R-CNN.
#     SSD etiketleri 1-based'dir (background=0 ayrilmis), Faster R-CNN ile ayni.

from typing import Optional

import numpy as np


# IoU helpers / IoU yardimcilari.

def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    # Compute IoU matrix between two sets of boxes.
    # Iki bbox seti arasinda IoU matrisi hesaplar.
    #
    # Args / Parametreler:
    #     boxes_a: shape (N, 4) in [x_min, y_min, x_max, y_max] format.
    #     boxes_a: shape (N, 4), [x_min, y_min, x_max, y_max] formatinda.
    #     boxes_b: shape (M, 4) in [x_min, y_min, x_max, y_max] format.
    #     boxes_b: shape (M, 4), [x_min, y_min, x_max, y_max] formatinda.
    #
    # Returns / Donus:
    #     IoU matrix of shape (N, M).
    #     (N, M) shape'inde IoU matrisi.
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Areas / Alanlar.
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Pairwise intersection / Cift olarak kesisim.
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


# TP/FP/FN counting / TP/FP/FN sayma.

def count_tp_fp_fn_per_class(
    preds:        list[dict],
    targets:      list[dict],
    score_threshold: float,
    iou_threshold:   float = 0.5,
) -> dict:
    # Count TP, FP, FN per class at the given confidence threshold.
    # Verilen confidence threshold'da per class TP, FP, FN sayar.
    #
    # Args / Parametreler:
    #     preds: list of dicts {image_id, boxes, labels, scores} per image.
    #     preds: per-image dict listesi {image_id, boxes, labels, scores}.
    #     targets: list of dicts {image_id, boxes, labels} per image.
    #     targets: per-image dict listesi {image_id, boxes, labels}.
    #
    # Returns / Donus:
    #     dict with keys "tp", "fp", "fn", each mapping label_id -> count.
    #     "tp", "fp", "fn" anahtarli dict, her biri label_id -> count.
    tp: dict[int, int] = {}
    fp: dict[int, int] = {}
    fn: dict[int, int] = {}

    # Match preds to targets by image_id.
    # Image_id'ye gore preds ile targets'i eslesfir.
    targets_by_id = {t["image_id"]: t for t in targets}

    for pred in preds:
        target = targets_by_id.get(pred["image_id"])
        if target is None:
            # No target for this image: every prediction above threshold is FP.
            # Bu goruntu icin target yok: threshold ustundeki her tahmin FP.
            mask = pred["scores"] >= score_threshold
            for lbl in pred["labels"][mask]:
                fp[int(lbl)] = fp.get(int(lbl), 0) + 1
            continue

        # Filter predictions by score threshold.
        # Tahminleri score threshold'a gore filtrele.
        mask = pred["scores"] >= score_threshold
        p_boxes  = pred["boxes"][mask]
        p_labels = pred["labels"][mask]

        gt_boxes  = target["boxes"]
        gt_labels = target["labels"]

        # Track which gt boxes are already matched.
        # Hangi gt bbox'larin zaten eslestirildigini takip et.
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        if len(p_boxes) > 0 and len(gt_boxes) > 0:
            iou = box_iou_matrix(p_boxes, gt_boxes)

            # Greedy matching by descending score (within this image).
            # Bu goruntude azalan score'a gore acgozlu eslesfirme.
            # Note: predictions in p_boxes are not sorted; we sort indices here.
            # Not: p_boxes'taki tahminler sirali degil; burada indexleri sort ediyoruz.
            scores_filtered = pred["scores"][mask]
            sort_idx = np.argsort(-scores_filtered)

            for i in sort_idx:
                pred_label = int(p_labels[i])

                # Find best matching unmatched gt with same label and IoU above threshold.
                # Ayni etiketli, eslesmemis ve IoU threshold uzerinde en iyi gt'yi bul.
                same_label = (gt_labels == pred_label)
                candidates = same_label & (~gt_matched) & (iou[i] >= iou_threshold)

                if candidates.any():
                    j = int(np.argmax(np.where(candidates, iou[i], -1)))
                    tp[pred_label] = tp.get(pred_label, 0) + 1
                    gt_matched[j] = True
                else:
                    fp[pred_label] = fp.get(pred_label, 0) + 1
        else:
            # No predictions at all, but there are targets -> all FN.
            # Hic tahmin yok, ama target var -> hepsi FN.
            for lbl in p_labels:
                fp[int(lbl)] = fp.get(int(lbl), 0) + 1

        # Unmatched GT boxes are FN.
        # Eslesmemis GT bbox'lari FN.
        for j, lbl in enumerate(gt_labels):
            if not gt_matched[j]:
                fn[int(lbl)] = fn.get(int(lbl), 0) + 1

    return {"tp": tp, "fp": fp, "fn": fn}


# F1 picker / F1 secici.

def _safe_pr_f1(tp: int, fp: int, fn: int) -> tuple:
    # Compute P, R, F1 with safe division.
    # Guvenli bolme ile P, R, F1 hesapla.
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_custom_pr(
    all_predictions: list[dict],
    all_targets:     list[dict],
    class_names:     list[str],
    iou_threshold:   float = 0.5,
    n_thresholds:    int = 50,
) -> tuple:
    # Sweep confidence thresholds and pick the one maximizing overall F1.
    # Confidence threshold'lari tara ve genel F1'i maksimize edeni sec.
    #
    # Args / Parametreler:
    #     class_names: list of class names (without background).
    #     class_names: sinif isimleri listesi (background haric).
    #     iou_threshold: IoU threshold for TP/FP decision.
    #     iou_threshold: TP/FP karari icin IoU threshold.
    #     n_thresholds: number of confidence values to try in [0.001, 0.999].
    #     n_thresholds: [0.001, 0.999] aralginda denenecek confidence sayisi.
    #
    # Returns / Donus:
    #     overall: dict with keys precision, recall, f1, threshold.
    #     overall: dict, anahtarlar precision, recall, f1, threshold.
    #     per_class: dict class_name -> {precision, recall} at the chosen threshold.
    #     per_class: dict class_name -> {precision, recall} secilen threshold'da.
    thresholds = np.linspace(0.001, 0.999, n_thresholds)

    best_overall = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "threshold": 0.0}
    best_per_class: dict[str, dict] = {
        name: {"precision": 0.0, "recall": 0.0}
        for name in class_names
    }
    best_threshold_per_class_pr: Optional[dict] = None  # Recorded at best_overall threshold.

    for thr in thresholds:
        counts = count_tp_fp_fn_per_class(
            preds=all_predictions,
            targets=all_targets,
            score_threshold=float(thr),
            iou_threshold=iou_threshold,
        )
        tp_dict = counts["tp"]
        fp_dict = counts["fp"]
        fn_dict = counts["fn"]

        # Overall counts.
        # Genel sayimlar.
        total_tp = sum(tp_dict.values())
        total_fp = sum(fp_dict.values())
        total_fn = sum(fn_dict.values())
        p, r, f1 = _safe_pr_f1(total_tp, total_fp, total_fn)

        if f1 > best_overall["f1"]:
            best_overall = {
                "precision": p,
                "recall":    r,
                "f1":        f1,
                "threshold": float(thr),
            }
            # Snapshot per-class P/R at this threshold.
            # Bu threshold'da per-class P/R'yi kaydet.
            snapshot = {}
            for k, name in enumerate(class_names):
                cls_label = k + 1  # +1 shift (SSD label)
                cls_tp = tp_dict.get(cls_label, 0)
                cls_fp = fp_dict.get(cls_label, 0)
                cls_fn = fn_dict.get(cls_label, 0)
                cp, cr, _ = _safe_pr_f1(cls_tp, cls_fp, cls_fn)
                snapshot[name] = {"precision": cp, "recall": cr}
            best_threshold_per_class_pr = snapshot

    if best_threshold_per_class_pr is not None:
        best_per_class = best_threshold_per_class_pr

    return best_overall, best_per_class

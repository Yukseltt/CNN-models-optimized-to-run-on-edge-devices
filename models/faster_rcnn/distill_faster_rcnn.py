# distill_faster_rcnn.py
#
# Response-based knowledge distillation for Faster R-CNN.
# Faster R-CNN icin response-tabanli bilgi damitma.
#
# Teacher (ResNet50) transfers classification and box knowledge to
# student (MobileNet) via proposal sharing.
# Teacher (ResNet50), proposal paylasimi ile classification ve box bilgisini
# student'a (MobileNet) aktarir.
#
# Mechanism / Mekanizma:
#     1. Teacher produces proposals and ROI predictions.
#     1. Teacher proposal ve ROI tahminleri uretir.
#     2. Student runs normal detection with its own proposals (detection loss).
#     2. Student kendi proposal'lari ile normal detection yapar (detection loss).
#     3. Student also runs ROI head on TEACHER's proposals (distillation).
#     3. Student ayrica TEACHER'in proposal'lari ile ROI head calistirir (distillation).
#     4. KL divergence on class logits + smooth L1 on box deltas.
#     4. Class logits'te KL divergence + box delta'larda smooth L1.
#
# Usage / Kullanim:
#     from distill_faster_rcnn import distill
#     distill(data_dir=..., cfg={...})

import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import (
    CocoDetectionDataset,
    build_train_transform,
    build_eval_transform,
    collate_fn,
)
from model_factory import create_faster_rcnn, count_parameters
from metrics_logger import build_excel, append_epoch_row
from eval_metrics import compute_custom_pr


# Path constants / Yol sabitleri.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR     = PROJECT_ROOT / "runs"


# Default distillation configuration / Varsayilan damitma konfigurasyonu.

DEFAULT_CFG = {
    "EPOCHS":         40,
    "BATCH_SIZE":     8,
    "PATIENCE":       15,
    "WORKERS":        4,
    "SEED":           0,
    "DEVICE":         "cuda:0",

    "LR0":            0.005,
    "LRF":            0.01,
    "MOMENTUM":       0.9,
    "WEIGHT_DECAY":   0.0005,
    "WARMUP_EPOCHS":  1,

    # Teacher and student backbones.
    # Teacher ve student backbone'lari.
    "TEACHER_BACKBONE":  "resnet50",
    "STUDENT_BACKBONE":  "mobilenet",

    # Checkpoints to warm-start from (None = build fresh).
    # Warm-start icin checkpoint'ler (None = sifirdan olustur).
    "TEACHER_CKPT":      None,
    "STUDENT_CKPT":      None,

    # Distillation hyperparameters.
    # Damitma hyperparametreleri.
    "TEMPERATURE":       3.0,
    "BETA_CLS":          0.7,
    "BETA_BOX":          0.5,

    "PROJECT":        str(RUNS_DIR),
    "NAME":           "faster_rcnn_distill_default",
    "OUTPUT_XLSX":    "training_metrics.xlsx",

    "LOG_INTERVAL":   50,
}


# Helpers / Yardimcilar.

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lr(epoch, total_epochs, warmup_epochs, lr0, lrf) -> float:
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lr0 * (lrf + (1 - lrf) * cosine)


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def resolve_run_dir(project_dir: str, base_name: str) -> Path:
    project = Path(project_dir)
    project.mkdir(parents=True, exist_ok=True)
    if not (project / base_name).exists():
        return project / base_name
    counter = 2
    while (project / f"{base_name}{counter}").exists():
        counter += 1
    return project / f"{base_name}{counter}"


# Dataloaders / Dataloader'lar.

def build_dataloaders(data_dir: str, cfg: dict):
    data_dir = Path(data_dir)
    train_ds = CocoDetectionDataset(
        images_dir=str(data_dir / "train" / "images"),
        annotation_path=str(data_dir / "train" / "_annotations.coco.json"),
        transform=build_train_transform(),
    )
    val_ds = CocoDetectionDataset(
        images_dir=str(data_dir / "val" / "images"),
        annotation_path=str(data_dir / "val" / "_annotations.coco.json"),
        transform=build_eval_transform(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True,
        num_workers=cfg["WORKERS"], collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
        num_workers=cfg["WORKERS"], collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader, train_ds.class_names, train_ds.num_classes


# Distillation core / Damitma cekirdegi.

def roi_forward_on_proposals(roi_heads, features, proposals, image_shapes):
    # Manually run the ROI head chain on given proposals.
    # Verilen proposal'lar uzerinde ROI head zincirini manuel calistirir.
    #
    # Chain / Zincir: box_roi_pool -> box_head -> box_predictor
    # Returns class_logits and box_regression.
    # class_logits ve box_regression doner.
    box_features = roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = roi_heads.box_head(box_features)
    class_logits, box_regression = roi_heads.box_predictor(box_features)
    return class_logits, box_regression


def distillation_losses(
    s_logits, s_boxes,
    t_logits, t_boxes,
    temperature: float,
) -> tuple:
    # Compute classification (KL) and box (smooth L1) distillation losses.
    # Classification (KL) ve box (smooth L1) damitma loss'larini hesaplar.
    #
    # Classification: temperature-scaled KL divergence.
    # Classification: temperature-olcekli KL divergence.
    T = temperature
    soft_teacher = F.softmax(t_logits / T, dim=1)
    log_soft_student = F.log_softmax(s_logits / T, dim=1)
    l_cls = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (T * T)

    # Box: smooth L1 between teacher and student box deltas.
    # Box: teacher ve student box delta'lari arasi smooth L1.
    l_box = F.smooth_l1_loss(s_boxes, t_boxes, reduction="mean")

    return l_cls, l_box


# Validation (same hybrid pass as training script).
# Validation (training scriptiyle ayni hybrid pass).

def predictions_to_coco_format(predictions, image_ids):
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        boxes  = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            coco_results.append({
                "image_id":    int(img_id),
                "category_id": int(label) - 1,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
            })
    return coco_results


def compute_per_class_map_coco(coco_eval, class_names):
    precision = coco_eval.eval["precision"]
    per_class = {}
    for k, name in enumerate(class_names):
        p50 = precision[0, :, k, 0, 2]
        p50 = p50[p50 > -1]
        ap50 = float(p50.mean()) if len(p50) > 0 else None
        p_all = precision[:, :, k, 0, 2]
        p_all = p_all[p_all > -1]
        ap_all = float(p_all.mean()) if len(p_all) > 0 else None
        per_class[name] = {"map50": ap50, "map50_95": ap_all}
    return per_class


def validate(model, loader, coco_gt, class_names, device):
    # Hybrid validation: losses (BN-frozen train mode) + predictions (eval mode).
    # Hybrid validation: loss (BN-dondurulmus train mode) + tahmin (eval mode).
    val_losses = {
        "loss_classifier": 0.0, "loss_box_reg": 0.0,
        "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0,
    }
    n_batches = 0
    all_coco_results = []
    all_predictions, all_targets = [], []

    for images, targets in loader:
        images_dev  = [img.to(device) for img in images]
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        set_bn_eval(model)
        with torch.no_grad():
            loss_dict = model(images_dev, targets_dev)
        for k, v in loss_dict.items():
            val_losses[k] += v.item()
        n_batches += 1

        model.eval()
        with torch.no_grad():
            predictions = model(images_dev)
        image_ids = [int(t["image_id"].item()) for t in targets]
        all_coco_results.extend(predictions_to_coco_format(predictions, image_ids))

        for pred, target in zip(predictions, targets):
            all_predictions.append({
                "image_id": int(target["image_id"].item()),
                "boxes":    pred["boxes"].cpu().numpy(),
                "labels":   pred["labels"].cpu().numpy(),
                "scores":   pred["scores"].cpu().numpy(),
            })
            all_targets.append({
                "image_id": int(target["image_id"].item()),
                "boxes":    target["boxes"].numpy(),
                "labels":   target["labels"].numpy(),
            })

    val_losses = {k: v / n_batches for k, v in val_losses.items()}

    map50, map50_95 = 0.0, 0.0
    per_class_map = {n: {"map50": None, "map50_95": None} for n in class_names}
    if all_coco_results:
        coco_dt = coco_gt.loadRes(all_coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50_95 = float(coco_eval.stats[0])
        map50    = float(coco_eval.stats[1])
        per_class_map = compute_per_class_map_coco(coco_eval, class_names)

    overall, per_class_pr = compute_custom_pr(
        all_predictions=all_predictions, all_targets=all_targets,
        class_names=class_names, iou_threshold=0.5,
    )

    per_class_combined = {}
    for name in class_names:
        m  = per_class_map.get(name, {"map50": None, "map50_95": None})
        pr = per_class_pr.get(name, {"precision": None, "recall": None})
        per_class_combined[name] = {**m, **pr}

    return {
        "losses": val_losses,
        "metrics": {
            "precision": overall["precision"],
            "recall":    overall["recall"],
            "f1":        overall["f1"],
            "map50":     map50,
            "map50_95":  map50_95,
        },
        "per_class": per_class_combined,
    }


# Main distillation entrypoint / Ana damitma giris noktasi.

def distill(data_dir: str, cfg: Optional[dict] = None) -> None:
    # Run response-based distillation from teacher to student.
    # Teacher'dan student'a response-tabanli damitma calistirir.
    c = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(c["SEED"])

    if c["DEVICE"].startswith("cuda") and not torch.cuda.is_available():
        print(f"[Distill] CUDA not available, using CPU.")
        c["DEVICE"] = "cpu"
    device = c["DEVICE"]

    print(f"[Distill] Teacher: {c['TEACHER_BACKBONE']}  Student: {c['STUDENT_BACKBONE']}")
    print(f"[Distill] Temperature={c['TEMPERATURE']}  "
          f"beta_cls={c['BETA_CLS']}  beta_box={c['BETA_BOX']}")
    print(f"[Distill] Batch={c['BATCH_SIZE']}  Epochs={c['EPOCHS']}  "
          f"LR0={c['LR0']}  Patience={c['PATIENCE']}")

    # Dataloaders.
    # Dataloader'lar.
    train_loader, val_loader, class_names, num_classes = build_dataloaders(data_dir, c)
    print(f"[Distill] Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"[Distill] Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print(f"[Distill] Classes: {class_names} (num_classes={num_classes} incl. bg)")

    # Build teacher (frozen).
    # Teacher olustur (donmus).
    teacher = create_faster_rcnn(
        backbone=c["TEACHER_BACKBONE"], num_classes=num_classes, pretrained="coco",
    )
    if c["TEACHER_CKPT"]:
        ckpt = torch.load(c["TEACHER_CKPT"], map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model"])
        print(f"[Distill] Teacher loaded from {c['TEACHER_CKPT']}")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Build student (warm start or fresh).
    # Student olustur (warm start veya sifirdan).
    student = create_faster_rcnn(
        backbone=c["STUDENT_BACKBONE"], num_classes=num_classes, pretrained="coco",
    )
    if c["STUDENT_CKPT"]:
        ckpt = torch.load(c["STUDENT_CKPT"], map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        print(f"[Distill] Student warm-started from {c['STUDENT_CKPT']}")
    student = student.to(device)

    t_params = count_parameters(teacher)
    s_params = count_parameters(student)
    print(f"[Distill] Teacher params: {t_params['total_millions']:.2f}M (frozen)")
    print(f"[Distill] Student params: {s_params['total_millions']:.2f}M (trainable)")

    # Optimizer (student only).
    # Optimizer (sadece student).
    optimizer = SGD(
        [p for p in student.parameters() if p.requires_grad],
        lr=c["LR0"], momentum=c["MOMENTUM"], weight_decay=c["WEIGHT_DECAY"],
    )

    # Output dir + Excel.
    # Cikti dizini + Excel.
    save_dir = resolve_run_dir(c["PROJECT"], c["NAME"])
    save_dir.mkdir(parents=True, exist_ok=True)
    excel_path = save_dir / c["OUTPUT_XLSX"]
    build_excel(excel_path, class_names)
    print(f"[Distill] Output dir: {save_dir}")

    # COCO GT for validation.
    # Validation icin COCO GT.
    val_gt_path = Path(data_dir) / "val" / "_annotations.coco.json"
    coco_gt = COCO(str(val_gt_path))

    best_map50 = -1.0
    best_epoch = -1
    patience_counter = 0

    print(f"\n[Distill] Starting distillation...\n")

    for epoch in range(c["EPOCHS"]):
        epoch_start = time.time()

        lr = compute_lr(epoch, c["EPOCHS"], c["WARMUP_EPOCHS"], c["LR0"], c["LRF"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        print(f"=== Epoch {epoch}/{c['EPOCHS'] - 1}  lr={lr:.6f} ===")

        # Training epoch.
        # Egitim epoch'u.
        student.train()
        loss_sums = {
            "detection": 0.0, "distill_cls": 0.0, "distill_box": 0.0, "total": 0.0,
        }
        n_batches = 0
        ep_t0 = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images_dev  = [img.to(device) for img in images]
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Teacher forward (no grad): proposals + ROI predictions.
            # Teacher forward (no grad): proposal + ROI tahminleri.
            with torch.no_grad():
                t_images, _ = teacher.transform(images_dev)
                t_features = teacher.backbone(t_images.tensors)
                t_proposals, _ = teacher.rpn(t_images, t_features)
                t_image_shapes = t_images.image_sizes
                t_logits, t_boxes = roi_forward_on_proposals(
                    teacher.roi_heads, t_features, t_proposals, t_image_shapes
                )

            # Student normal detection loss (own proposals).
            # Student normal detection loss (kendi proposal'lari).
            loss_dict = student(images_dev, targets_dev)
            l_detection = sum(loss_dict.values())

            # Student ROI on teacher's proposals (distillation).
            # Teacher'in proposal'lari ile student ROI (distillation).
            s_images, _ = student.transform(images_dev)
            s_features = student.backbone(s_images.tensors)
            s_image_shapes = s_images.image_sizes
            s_logits, s_boxes = roi_forward_on_proposals(
                student.roi_heads, s_features, t_proposals, s_image_shapes
            )

            # Distillation losses.
            # Damitma loss'lari.
            l_cls, l_box = distillation_losses(
                s_logits, s_boxes, t_logits, t_boxes, c["TEMPERATURE"]
            )

            # Total loss.
            # Toplam loss.
            loss = l_detection + c["BETA_CLS"] * l_cls + c["BETA_BOX"] * l_box

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sums["detection"]   += l_detection.item()
            loss_sums["distill_cls"] += l_cls.item()
            loss_sums["distill_box"] += l_box.item()
            loss_sums["total"]       += loss.item()
            n_batches += 1

            if (batch_idx + 1) % c["LOG_INTERVAL"] == 0:
                elapsed = time.time() - ep_t0
                print(f"  [Epoch {epoch}] batch {batch_idx + 1}/{len(train_loader)}  "
                      f"det={loss_sums['detection']/n_batches:.4f}  "
                      f"d_cls={loss_sums['distill_cls']/n_batches:.4f}  "
                      f"d_box={loss_sums['distill_box']/n_batches:.4f}  "
                      f"elapsed={elapsed:.1f}s")

        train_losses_avg = {k: v / n_batches for k, v in loss_sums.items()}
        print(f"  Train: det={train_losses_avg['detection']:.4f}  "
              f"d_cls={train_losses_avg['distill_cls']:.4f}  "
              f"d_box={train_losses_avg['distill_box']:.4f}  "
              f"total={train_losses_avg['total']:.4f}")

        # Validation.
        # Validation.
        val_results = validate(student, val_loader, coco_gt, class_names, device)
        val_losses  = val_results["losses"]
        val_metrics = val_results["metrics"]
        per_class   = val_results["per_class"]

        print(f"  Val:   P={val_metrics['precision']:.4f}  "
              f"R={val_metrics['recall']:.4f}  F1={val_metrics['f1']:.4f}  "
              f"mAP@0.5={val_metrics['map50']:.4f}  "
              f"mAP@0.5:0.95={val_metrics['map50_95']:.4f}")

        # Excel logging (reuse detection loss format).
        # Excel logging (detection loss formatini tekrar kullan).
        # We log student's own detection losses for train, distill losses go to console.
        # Train icin student'in kendi detection loss'larini logla, distill loss'lar konsola.
        append_epoch_row(
            output_path=excel_path,
            epoch=epoch,
            lr=lr,
            train_losses={
                "loss_classifier":  0.0,
                "loss_box_reg":     0.0,
                "loss_objectness":  0.0,
                "loss_rpn_box_reg": train_losses_avg["detection"],
            },
            val_losses=val_losses,
            val_metrics=val_metrics,
            per_class_metrics=per_class,
            class_names=class_names,
        )

        # Save last.
        # last kaydet.
        torch.save(
            {
                "epoch": epoch, "model": student.state_dict(),
                "optimizer": optimizer.state_dict(), "cfg": c,
                "class_names": class_names,
            },
            save_dir / "last.pt",
        )

        # Save best.
        # best kaydet.
        if val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch, "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(), "cfg": c,
                    "class_names": class_names, "metric": val_metrics["map50"],
                },
                save_dir / "best.pt",
            )
            print(f"  >> NEW BEST mAP@0.5={best_map50:.4f} at epoch {best_epoch}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{c['PATIENCE']})  "
                  f"best={best_map50:.4f} @ epoch {best_epoch}")

        print(f"  Epoch time: {time.time() - epoch_start:.1f}s\n")

        if patience_counter >= c["PATIENCE"]:
            print(f"[Distill] Early stopping at epoch {epoch}. "
                  f"Best: epoch {best_epoch}, mAP@0.5={best_map50:.4f}")
            break

    print(f"\n[Distill] Distillation finished.")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Best mAP@0.5: {best_map50:.4f}")
    print(f"  Output dir:   {save_dir}")
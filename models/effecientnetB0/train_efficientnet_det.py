# train_efficientnet_det.py
#
# EfficientNet-B0 + FPN Faster R-CNN training script with custom training loop.
# Custom training loop ile EfficientNet-B0 + FPN Faster R-CNN training script.
#
# faster_rcnn/train_faster_rcnn.py ile ayni yapidadir; tek fark model factory'nin
# create_efficientnet_fpn_detector olmasi ve MIN_SIZE/MAX_SIZE ile native 640px
# cozunurlukte calismasidir. Loss anahtarlari Faster R-CNN ile ayni oldugundan
# train/val dongusu degismeden kullanilir.
# Same structure as faster_rcnn/train_faster_rcnn.py; only the model factory
# (create_efficientnet_fpn_detector) and the MIN_SIZE/MAX_SIZE native-640 resize
# differ. Loss keys match Faster R-CNN, so the train/val loop is unchanged.
#
# Components / Bilesenler:
#     - Manual warmup + cosine LR schedule
#     - Manuel warmup + cosine LR schedule
#     - Hybrid validation (val loss via BN-eval mode + predictions via full eval)
#     - Hybrid validation (BN-eval mode ile val loss + tam eval ile tahminler)
#     - pycocotools mAP + custom YOLO-style precision / recall / F1
#     - pycocotools mAP + custom YOLO tarzi precision / recall / F1
#     - Per-class metrics, Excel logging, best/last checkpointing, early stopping
#     - Per-class metrikler, Excel logging, best/last checkpoint, early stopping
#     - Resume from last.pt
#     - last.pt'den resume
#
# Usage / Kullanim:
#     from train_efficientnet_det import train
#     train(data_dir="/path/to/coco_root", cfg={"NAME": "...", ...})
#
# Resume usage / Resume kullanim:
#     train(data_dir=..., cfg={..., "RESUME_FROM": "runs_new/.../last.pt"})

import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim import SGD

# Bu host'ta DataLoader worker'lari varsayilan "file_descriptor" paylasimi ile
# EMFILE (too many open files) veriyor. "file_system" stratejisi sart.
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import (
    CocoDetectionDataset,
    build_train_transform,
    build_eval_transform,
    collate_fn,
)
from detection_model_factory import create_efficientnet_fpn_detector, count_parameters
from metrics_logger import build_excel, append_epoch_row
from eval_metrics import compute_custom_pr


# Path constants / Yol sabitleri.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR     = PROJECT_ROOT / "runs"


# Default training configuration / Varsayilan egitim konfigurasyonu.

DEFAULT_CFG = {
    "EPOCHS":         50,
    "BATCH_SIZE":     8,
    "PATIENCE":       10,
    "WORKERS":        4,
    "SEED":           0,
    "DEVICE":         "cuda:0",

    "LR0":            0.005,
    "LRF":            0.01,
    "MOMENTUM":       0.9,
    "WEIGHT_DECAY":   0.0005,
    "WARMUP_EPOCHS":  1,

    # Backbone EfficientNet-B0; "imagenet" -> backbone ImageNet1K, head'ler sifirdan.
    # Backbone EfficientNet-B0; "imagenet" -> ImageNet1K backbone, heads from scratch.
    "PRETRAINED":     "imagenet",
    # None -> tum backbone egitilebilir (termal domain'e adaptasyon icin onerilen).
    # None -> whole backbone trainable (recommended for thermal domain shift).
    "TRAINABLE_BACKBONE_LAYERS": None,

    # Dataset goruntuleri 640x512 / 640x640 -> native cozunurluk, upscale yok.
    # Dataset images are 640x512 / 640x640 -> native resolution, no upscaling.
    "MIN_SIZE":       640,
    "MAX_SIZE":       640,

    "PROJECT":        str(RUNS_DIR),
    "NAME":           "efficientnet_b0_fpn_fasterrcnn_default",
    "OUTPUT_XLSX":    "training_metrics.xlsx",

    "LOG_INTERVAL":   50,

    # If set to a path, resume training from that checkpoint.
    # Bir yola ayarlanirsa, o checkpoint'ten egitime devam eder.
    "RESUME_FROM":    None,
}


# Loss keys for Faster R-CNN / Faster R-CNN icin loss anahtarlari.
FRCNN_LOSS_KEYS = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)


# Helpers / Yardimcilar.

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lr(
    epoch:         int,
    total_epochs:  int,
    warmup_epochs: int,
    lr0:           float,
    lrf:           float,
) -> float:
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


def _empty_loss_dict() -> dict:
    return {k: 0.0 for k in FRCNN_LOSS_KEYS}


# Dataloader factory / Dataloader factory.

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
        train_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["WORKERS"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.class_names, train_ds.num_classes


# Train epoch / Train epoch.

def train_one_epoch(
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    loader:       DataLoader,
    device:       str,
    epoch:        int,
    log_interval: int = 50,
) -> dict:
    model.train()
    loss_sums = _empty_loss_dict()
    n_batches = 0
    epoch_start = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            loss_sums[k] += v.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - epoch_start
            avg_loss = sum(loss_sums.values()) / n_batches
            print(
                f"  [Epoch {epoch}] batch {batch_idx + 1}/{len(loader)}  "
                f"avg_loss={avg_loss:.4f}  elapsed={elapsed:.1f}s"
            )

    return {k: v / n_batches for k, v in loss_sums.items()}


# Validation helpers / Validation yardimcilari.

def predictions_to_coco_format(predictions: list, image_ids: list) -> list:
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        boxes  = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            coco_results.append({
                "image_id":    int(img_id),
                "category_id": int(label) - 1,
                "bbox": [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ],
                "score":       float(score),
            })
    return coco_results


def compute_per_class_map_coco(coco_eval: COCOeval, class_names: list) -> dict:
    precision = coco_eval.eval["precision"]
    per_class: dict = {}

    for k, name in enumerate(class_names):
        p_50 = precision[0, :, k, 0, 2]
        p_50 = p_50[p_50 > -1]
        ap_50 = float(p_50.mean()) if len(p_50) > 0 else None

        p_all = precision[:, :, k, 0, 2]
        p_all = p_all[p_all > -1]
        ap_all = float(p_all.mean()) if len(p_all) > 0 else None

        per_class[name] = {
            "map50":    ap_50,
            "map50_95": ap_all,
        }
    return per_class


# Validation loop / Validation loop.

def validate(
    model:       nn.Module,
    loader:      DataLoader,
    coco_gt:     COCO,
    class_names: list,
    device:      str,
) -> dict:
    val_losses = _empty_loss_dict()
    n_batches = 0

    all_coco_results = []
    all_predictions: list = []
    all_targets:     list = []

    val_start = time.time()

    for images, targets in loader:
        images_dev  = [img.to(device) for img in images]
        targets_dev = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        # Pass 1: validation loss (train mode but BN frozen).
        # Pass 1: validation loss (train mode ama BN dondurulmus).
        model.train()
        set_bn_eval(model)
        with torch.no_grad():
            loss_dict = model(images_dev, targets_dev)
        for k, v in loss_dict.items():
            val_losses[k] += v.item()
        n_batches += 1

        # Pass 2: predictions (full eval mode).
        # Pass 2: tahminler (tam eval mode).
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

    map50    = 0.0
    map50_95 = 0.0
    per_class_map: dict = {name: {"map50": None, "map50_95": None} for name in class_names}

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
        all_predictions=all_predictions,
        all_targets=all_targets,
        class_names=class_names,
        iou_threshold=0.5,
    )

    per_class_combined: dict = {}
    for name in class_names:
        m  = per_class_map.get(name, {"map50": None, "map50_95": None})
        pr = per_class_pr.get(name, {"precision": None, "recall": None})
        per_class_combined[name] = {**m, **pr}

    elapsed = time.time() - val_start
    print(f"  Validation done in {elapsed:.1f}s")

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


# Resume helpers / Resume yardimcilari.

def load_resume_state(
    resume_path: str,
    model:       nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      str,
) -> dict:
    # Bir checkpoint'ten model ve optimizer state'ini yukler.
    # Load model and optimizer state from a checkpoint.
    print(f"[EFFB0-DET] Resuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = int(ckpt["epoch"]) + 1
    print(f"[EFFB0-DET] Resumed at epoch {start_epoch} "
          f"(checkpoint was saved at epoch {ckpt['epoch']}).")

    return {
        "start_epoch":        start_epoch,
        "checkpoint_cfg":     ckpt.get("cfg", {}),
        "checkpoint_classes": ckpt.get("class_names", []),
    }


def load_best_state(best_path: Path) -> dict:
    # En iyi mAP ve en iyi epoch'u almak icin best.pt'yi okur.
    # Read best.pt to get best mAP and best epoch.
    if not best_path.exists():
        return {"best_map50": -1.0, "best_epoch": -1}
    bckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    return {
        "best_map50": float(bckpt.get("metric", -1.0)),
        "best_epoch": int(bckpt.get("epoch", -1)),
    }


# Main entrypoint / Ana giris noktasi.

def train(data_dir: str, cfg: Optional[dict] = None) -> None:
    c = {**DEFAULT_CFG, **(cfg or {})}

    set_seed(c["SEED"])

    if c["DEVICE"].startswith("cuda") and not torch.cuda.is_available():
        print(f"[EFFB0-DET] CUDA not available, falling back to CPU.")
        c["DEVICE"] = "cpu"

    is_resume = c.get("RESUME_FROM") is not None

    print(f"[EFFB0-DET] Backbone:  efficientnet_b0 + FPN (FasterRCNN head)")
    print(f"[EFFB0-DET] Pretrained:{c['PRETRAINED']}  "
          f"trainable_backbone_layers={c['TRAINABLE_BACKBONE_LAYERS']}")
    print(f"[EFFB0-DET] Device:    {c['DEVICE']}")
    print(f"[EFFB0-DET] Batch:     {c['BATCH_SIZE']}, Epochs: {c['EPOCHS']}")
    print(f"[EFFB0-DET] Resize:    min_size={c['MIN_SIZE']} max_size={c['MAX_SIZE']}")
    print(f"[EFFB0-DET] LR0:       {c['LR0']}, LRF: {c['LRF']}")
    print(f"[EFFB0-DET] Patience:  {c['PATIENCE']}")
    if is_resume:
        print(f"[EFFB0-DET] Resume:    {c['RESUME_FROM']}")

    train_loader, val_loader, class_names, num_classes = build_dataloaders(data_dir, c)
    print(f"[EFFB0-DET] Train: {len(train_loader.dataset)} images, "
          f"{len(train_loader)} batches")
    print(f"[EFFB0-DET] Val:   {len(val_loader.dataset)} images, "
          f"{len(val_loader)} batches")
    print(f"[EFFB0-DET] Classes: {class_names} (num_classes={num_classes} incl. bg)")

    model = create_efficientnet_fpn_detector(
        num_classes=num_classes,
        pretrained=c["PRETRAINED"],
        min_size=c["MIN_SIZE"],
        max_size=c["MAX_SIZE"],
        trainable_backbone_layers=c["TRAINABLE_BACKBONE_LAYERS"],
    )
    model = model.to(c["DEVICE"])
    params = count_parameters(model)
    print(f"[EFFB0-DET] Total params: {params['total_millions']:.2f}M "
          f"(trainable {params['trainable_millions']:.2f}M)")

    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=c["LR0"],
        momentum=c["MOMENTUM"],
        weight_decay=c["WEIGHT_DECAY"],
    )

    # Resume mode: reuse existing run dir, skip Excel rebuild.
    # Resume mode: mevcut run dizinini yeniden kullan, Excel'i yeniden olusturma.
    start_epoch      = 0
    best_map50       = -1.0
    best_epoch       = -1
    patience_counter = 0

    if is_resume:
        save_dir = Path(c["RESUME_FROM"]).resolve().parent
        excel_path = save_dir / c["OUTPUT_XLSX"]
        if not excel_path.exists():
            raise FileNotFoundError(
                f"Excel not found in resume dir: {excel_path}. "
                f"Cannot resume without existing metrics file."
            )
        print(f"[EFFB0-DET] Resume dir: {save_dir}")

        resume_state = load_resume_state(c["RESUME_FROM"], model, optimizer, c["DEVICE"])
        start_epoch = resume_state["start_epoch"]

        best_state = load_best_state(save_dir / "best.pt")
        best_map50 = best_state["best_map50"]
        best_epoch = best_state["best_epoch"]
        patience_counter = max(0, (start_epoch - 1) - best_epoch)
        print(f"[EFFB0-DET] Best so far: mAP@0.5={best_map50:.4f} @ epoch {best_epoch}")
        print(f"[EFFB0-DET] Patience counter: {patience_counter}/{c['PATIENCE']}")
    else:
        save_dir = resolve_run_dir(c["PROJECT"], c["NAME"])
        save_dir.mkdir(parents=True, exist_ok=True)
        excel_path = save_dir / c["OUTPUT_XLSX"]
        print(f"[EFFB0-DET] Output dir: {save_dir}")
        build_excel(excel_path, class_names)

    val_gt_path = Path(data_dir) / "val" / "_annotations.coco.json"
    coco_gt = COCO(str(val_gt_path))

    print(f"\n[EFFB0-DET] Starting training from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, c["EPOCHS"]):
        epoch_start = time.time()

        lr = compute_lr(
            epoch=epoch,
            total_epochs=c["EPOCHS"],
            warmup_epochs=c["WARMUP_EPOCHS"],
            lr0=c["LR0"],
            lrf=c["LRF"],
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        print(f"=== Epoch {epoch}/{c['EPOCHS'] - 1}  lr={lr:.6f} ===")

        train_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=c["DEVICE"],
            epoch=epoch,
            log_interval=c["LOG_INTERVAL"],
        )
        train_total = sum(train_losses.values())
        print(f"  Train losses: cls={train_losses['loss_classifier']:.4f}  "
              f"box={train_losses['loss_box_reg']:.4f}  "
              f"obj={train_losses['loss_objectness']:.4f}  "
              f"rpn={train_losses['loss_rpn_box_reg']:.4f}  "
              f"total={train_total:.4f}")

        val_results = validate(
            model=model,
            loader=val_loader,
            coco_gt=coco_gt,
            class_names=class_names,
            device=c["DEVICE"],
        )
        val_losses  = val_results["losses"]
        val_metrics = val_results["metrics"]
        per_class   = val_results["per_class"]
        val_total   = sum(val_losses.values())

        print(f"  Val losses:   total={val_total:.4f}")
        print(f"  Val metrics:  P={val_metrics['precision']:.4f}  "
              f"R={val_metrics['recall']:.4f}  "
              f"F1={val_metrics['f1']:.4f}  "
              f"mAP@0.5={val_metrics['map50']:.4f}  "
              f"mAP@0.5:0.95={val_metrics['map50_95']:.4f}")

        append_epoch_row(
            output_path=excel_path,
            epoch=epoch,
            lr=lr,
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_metrics,
            per_class_metrics=per_class,
            class_names=class_names,
        )

        torch.save(
            {
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "cfg":         c,
                "class_names": class_names,
            },
            save_dir / "last.pt",
        )

        improved = val_metrics["map50"] > best_map50
        if improved:
            best_map50 = val_metrics["map50"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "cfg":         c,
                    "class_names": class_names,
                    "metric":      val_metrics["map50"],
                },
                save_dir / "best.pt",
            )
            print(f"  >> NEW BEST mAP@0.5={best_map50:.4f} at epoch {best_epoch}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{c['PATIENCE']})  "
                  f"best mAP@0.5={best_map50:.4f} @ epoch {best_epoch}")

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s\n")

        if patience_counter >= c["PATIENCE"]:
            print(f"[EFFB0-DET] Early stopping at epoch {epoch}. "
                  f"Best: epoch {best_epoch}, mAP@0.5={best_map50:.4f}")
            break

    print(f"\n[EFFB0-DET] Training finished.")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Best mAP@0.5: {best_map50:.4f}")
    print(f"  Output dir:   {save_dir}")

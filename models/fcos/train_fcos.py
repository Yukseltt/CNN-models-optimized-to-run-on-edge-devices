# train_fcos.py
#
# FCOS (ResNet50 + FPN) training script with custom training loop.
# Custom training loop ile FCOS (ResNet50 + FPN) training script.
#
# Components / Bilesenler:
#     - Manual warmup + cosine LR schedule
#     - Manuel warmup + cosine LR schedule
#     - Hybrid validation (val loss via BN-eval mode + predictions via full eval)
#     - Hybrid validation (BN-eval mode ile val loss + tam eval ile tahminler)
#     - pycocotools mAP computation
#     - pycocotools mAP hesabi
#     - Custom YOLO-style precision / recall / F1
#     - Custom YOLO tarzi precision / recall / F1
#     - Per-class metrics, Excel logging, best/last checkpointing, early stopping
#     - Per-class metrikler, Excel logging, best/last checkpoint, early stopping
#     - Resume from last.pt
#     - last.pt'den resume
#
# FCOS-specific notes / FCOS'ya ozel notlar:
#     - Loss dict has THREE components: "bbox_regression", "classification",
#       and "bbox_ctrness" (centerness loss for low-quality filtering).
#     - Loss dict UC bilesenli: "bbox_regression", "classification",
#       ve "bbox_ctrness" (dusuk-kaliteli filtreleme icin centerness loss).
#     - FCOS uses BatchNorm in FPN/head; BN-eval trick still applies for val.
#     - FCOS FPN/head'inde BatchNorm var; val icin BN-eval trick yine gecerli.
#
# Usage / Kullanim:
#     from train_fcos import train
#     train(data_dir="/path/to/coco_root", cfg={"BACKBONE": "resnet50", ...})
#
# Resume usage / Resume kullanim:
#     train(data_dir=..., cfg={..., "RESUME_FROM": "runs/.../last.pt"})

import math
import random
import resource
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


# Multiprocessing setup / Multiprocessing kurulum.
#
# Default "file_descriptor" sharing strategy passes one FD per shared tensor
# between workers and the main process. With persistent_workers and
# prefetch_factor > 1 this exhausts the per-process FD limit quickly and
# raises "Too many open files (24)" + secondary "Pin memory thread exited"
# errors. "file_system" uses /dev/shm filenames instead -> no FD accumulation.
# Varsayilan "file_descriptor" stratejisi, paylasilan her tensor icin
# worker-main arasinda bir FD tasir. persistent_workers + prefetch_factor > 1
# ile per-process FD limiti hizla dolar ve "Too many open files (24)" +
# arkasindan "Pin memory thread exited" hatasi olusur. "file_system",
# FD yerine /dev/shm dosya adi kullanir -> FD birikimi olmaz.
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# Best-effort raise of the per-process FD soft limit up to the hard limit.
# If the user has root or sufficient capabilities this helps the file_system
# strategy's tmpfs entries too.
# Per-process FD soft limit'ini hard limit'e cikarmaya calis. Root veya yetki
# varsa, file_system stratejisinin tmpfs girisleri icin de yardim eder.
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
except (ValueError, OSError):
    pass
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
from model_factory import create_fcos, count_parameters
from metrics_logger import build_excel, append_epoch_row
from eval_metrics import compute_custom_pr


# Path constants / Yol sabitleri.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR     = PROJECT_ROOT / "runs"


# Default training configuration / Varsayilan egitim konfigurasyonu.

DEFAULT_CFG = {
    "EPOCHS":         50,
    "BATCH_SIZE":     64,
    "PATIENCE":       10,
    # MIG slice + shared CPU host -> stay below physical core count.
    # MIG dilimi + paylasimli CPU -> fiziksel core sayisinin altinda kal.
    "WORKERS":        4,
    "PREFETCH":       2,
    "SEED":           0,
    "DEVICE":         "cuda:0",

    "LR0":            0.01,
    "LRF":            0.01,
    "MOMENTUM":       0.9,
    "WEIGHT_DECAY":   0.00004,
    "WARMUP_EPOCHS":  1,
    # Iteration-level linear warmup over the first epoch. FCOS diverges to
    # NaN if it sees full LR from batch 1 (freshly-init head + SGD momentum),
    # so we ramp LR from WARMUP_FACTOR*lr up to lr over WARMUP_ITERS steps.
    # Ilk epoch boyunca iterasyon-seviyesi linear warmup. FCOS, ilk batch'ten
    # tam LR gorurse NaN'a diverge eder (yeni init head + SGD momentum), bu
    # yuzden LR'yi WARMUP_ITERS adimda WARMUP_FACTOR*lr'den lr'ye rampalariz.
    "WARMUP_ITERS":   1000,
    "WARMUP_FACTOR":  0.001,
    # Gradient clipping as a safety net against loss spikes -> NaN.
    # Loss sicramalarina (-> NaN) karsi guvenlik agi olarak gradient clipping.
    "MAX_GRAD_NORM":  10.0,

    "BACKBONE":       "resnet50",
    "PRETRAINED":     True,

    # Mixed-precision settings / Mixed-precision ayarlari.
    # bf16 on Hopper (H200) gives big speedup without GradScaler.
    # Hopper'da (H200) bf16 GradScaler olmadan buyuk speedup verir.
    "USE_AMP":        True,
    "AMP_DTYPE":      "bfloat16",   # "bfloat16" or "float16"
    "CUDNN_BENCHMARK": True,

    "PROJECT":        str(RUNS_DIR),
    "NAME":           "fcos_resnet50_default",
    "OUTPUT_XLSX":    "training_metrics.xlsx",

    "LOG_INTERVAL":   50,

    # If set to a path, resume training from that checkpoint.
    # Bir yola ayarlanirsa, o checkpoint'ten egitime devam eder.
    "RESUME_FROM":    None,
}


# Loss keys for SSD / SSD icin loss anahtarlari.
FCOS_LOSS_KEYS = ("bbox_regression", "classification", "bbox_ctrness")


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
    return {k: 0.0 for k in FCOS_LOSS_KEYS}


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

    # persistent_workers + prefetch_factor keep workers alive between epochs and
    # buffer more batches, eliminating worker startup overhead and reducing the
    # chance of GPU starvation when augmentation is light.
    # persistent_workers + prefetch_factor, worker'lari epoch'lar arasi canli
    # tutar ve daha fazla batch buffer'lar; worker startup maliyetini siler ve
    # hafif augmentation'da GPU starvation ihtimalini azaltir.
    workers   = cfg["WORKERS"]
    prefetch  = cfg.get("PREFETCH", 4)
    dl_extra  = (
        {"persistent_workers": True, "prefetch_factor": prefetch}
        if workers > 0 else {}
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        **dl_extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **dl_extra,
    )

    return train_loader, val_loader, train_ds.class_names, train_ds.num_classes


# Train epoch / Train epoch.

def train_one_epoch(
    model:         nn.Module,
    optimizer:     torch.optim.Optimizer,
    loader:        DataLoader,
    device:        str,
    epoch:         int,
    log_interval:  int = 50,
    amp_ctx=None,
    scaler:        Optional[torch.amp.GradScaler] = None,
    base_lr:       Optional[float] = None,
    warmup_iters:  int = 0,
    warmup_factor: float = 0.001,
    max_grad_norm: Optional[float] = None,
) -> dict:
    model.train()
    loss_sums = _empty_loss_dict()
    n_batches = 0
    skipped   = 0
    epoch_start = time.time()

    # Default to a no-op autocast context if AMP is disabled.
    # AMP kapaliysa no-op autocast baglami kullan.
    if amp_ctx is None:
        amp_ctx = torch.amp.autocast(device_type="cuda", enabled=False)

    # Iteration-level warmup is only applied during the first epoch.
    # Iterasyon-seviyesi warmup sadece ilk epoch'ta uygulanir.
    do_warmup = epoch == 0 and warmup_iters > 0 and base_lr is not None

    for batch_idx, (images, targets) in enumerate(loader):
        # Linear LR warmup: ramp from warmup_factor*base_lr up to base_lr.
        # Linear LR warmup: warmup_factor*base_lr'den base_lr'ye rampala.
        if do_warmup and batch_idx < warmup_iters:
            alpha  = batch_idx / warmup_iters
            factor = warmup_factor * (1.0 - alpha) + alpha
            for g in optimizer.param_groups:
                g["lr"] = base_lr * factor

        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

        with amp_ctx:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        # Skip non-finite batches so a single bad step cannot poison the
        # weights (and propagate NaN to every subsequent batch).
        # Tek bir bozuk adimin agirliklari zehirlemesini (ve NaN'i sonraki
        # tum batch'lere yaymasini) onlemek icin non-finite batch'leri atla.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            if skipped <= 5 or skipped % 50 == 0:
                print(f"  [Epoch {epoch}] batch {batch_idx + 1}: "
                      f"non-finite loss, skipped ({skipped} total)")
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        for k in loss_sums:
            if k in loss_dict:
                loss_sums[k] += loss_dict[k].item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - epoch_start
            avg_loss = sum(loss_sums.values()) / max(1, n_batches)
            print(
                f"  [Epoch {epoch}] batch {batch_idx + 1}/{len(loader)}  "
                f"avg_loss={avg_loss:.4f}  elapsed={elapsed:.1f}s"
            )

    return {k: v / max(1, n_batches) for k, v in loss_sums.items()}


# Validation helpers / Validation yardimcilari.

def predictions_to_coco_format(predictions: list, image_ids: list) -> list:
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        # .float() first: under bf16 autocast the head emits bfloat16 boxes/
        # scores, and numpy() has no bfloat16 support (raises TypeError).
        # Once .float(): bf16 autocast altinda head bfloat16 box/score uretir,
        # numpy() bfloat16 desteklemez (TypeError verir).
        boxes  = pred["boxes"].float().cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].float().cpu().numpy()

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
    amp_ctx=None,
) -> dict:
    val_losses = _empty_loss_dict()
    n_batches = 0

    all_coco_results = []
    all_predictions: list = []
    all_targets:     list = []

    val_start = time.time()

    if amp_ctx is None:
        amp_ctx = torch.amp.autocast(device_type="cuda", enabled=False)

    for images, targets in loader:
        images_dev  = [img.to(device, non_blocking=True) for img in images]
        targets_dev = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()}
            for t in targets
        ]

        # Pass 1: validation loss (train mode but BN frozen).
        # Pass 1: validation loss (train mode ama BN dondurulmus).
        model.train()
        set_bn_eval(model)
        with torch.no_grad(), amp_ctx:
            loss_dict = model(images_dev, targets_dev)
        for k in val_losses:
            if k in loss_dict:
                val_losses[k] += loss_dict[k].item()
        n_batches += 1

        # Pass 2: predictions (full eval mode).
        # Pass 2: tahminler (tam eval mode).
        model.eval()
        with torch.no_grad(), amp_ctx:
            predictions = model(images_dev)

        image_ids = [int(t["image_id"].item()) for t in targets]
        all_coco_results.extend(predictions_to_coco_format(predictions, image_ids))

        for pred, target in zip(predictions, targets):
            all_predictions.append({
                "image_id": int(target["image_id"].item()),
                "boxes":    pred["boxes"].float().cpu().numpy(),
                "labels":   pred["labels"].cpu().numpy(),
                "scores":   pred["scores"].float().cpu().numpy(),
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
    # Load model and optimizer state from a checkpoint.
    # Bir checkpoint'ten model ve optimizer state'ini yukler.
    print(f"[FCOS] Resuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = int(ckpt["epoch"]) + 1
    print(f"[FCOS] Resumed at epoch {start_epoch} "
          f"(checkpoint was saved at epoch {ckpt['epoch']}).")

    return {
        "start_epoch":        start_epoch,
        "checkpoint_cfg":     ckpt.get("cfg", {}),
        "checkpoint_classes": ckpt.get("class_names", []),
    }


def load_best_state(best_path: Path) -> dict:
    # Read best.pt to get best mAP and best epoch.
    # En iyi mAP ve en iyi epoch'u almak icin best.pt'yi okur.
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
        print(f"[FCOS] CUDA not available, falling back to CPU.")
        c["DEVICE"] = "cpu"

    is_resume = c.get("RESUME_FROM") is not None

    # cuDNN autotune: FCOS's internal transform reshapes inputs to a fixed
    # 320x320, so shapes are constant -> autotune is a free win.
    # cuDNN autotune: FCOS'in dahili transform'u girdiyi sabit 320x320'ye
    # yeniden bicimlendirir, sekiller sabit -> autotune bedava kazanc.
    if c.get("CUDNN_BENCHMARK", True) and c["DEVICE"].startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # AMP autocast context. bf16 needs no GradScaler on Hopper (H200).
    # AMP autocast baglami. Hopper'da (H200) bf16 GradScaler gerektirmez.
    use_amp   = bool(c.get("USE_AMP", True)) and c["DEVICE"].startswith("cuda")
    amp_dtype = torch.bfloat16 if c.get("AMP_DTYPE", "bfloat16") == "bfloat16" else torch.float16
    amp_ctx   = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else
        torch.amp.autocast(device_type="cuda", enabled=False)
    )
    # GradScaler only required for float16; bfloat16 has fp32-equivalent range.
    # GradScaler sadece float16 icin gerekli; bfloat16 fp32 menziline sahip.
    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp and amp_dtype is torch.float16 else
        None
    )

    print(f"[FCOS] Backbone:  {c['BACKBONE']} (FCOS (ResNet50 + FPN))")
    print(f"[FCOS] Device:    {c['DEVICE']}")
    print(f"[FCOS] Batch:     {c['BATCH_SIZE']}, Epochs: {c['EPOCHS']}, Workers: {c['WORKERS']}")
    print(f"[FCOS] LR0:       {c['LR0']}, LRF: {c['LRF']}")
    print(f"[FCOS] Patience:  {c['PATIENCE']}")
    print(f"[FCOS] AMP:       {use_amp} (dtype={amp_dtype if use_amp else 'fp32'})  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    try:
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[FCOS] MP share:  {mp.get_sharing_strategy()}  RLIMIT_NOFILE: soft={_soft}, hard={_hard}")
    except (ValueError, OSError):
        pass
    if is_resume:
        print(f"[FCOS] Resume:    {c['RESUME_FROM']}")

    train_loader, val_loader, class_names, num_classes = build_dataloaders(data_dir, c)
    print(f"[FCOS] Train: {len(train_loader.dataset)} images, "
          f"{len(train_loader)} batches")
    print(f"[FCOS] Val:   {len(val_loader.dataset)} images, "
          f"{len(val_loader)} batches")
    print(f"[FCOS] Classes: {class_names} (num_classes={num_classes} incl. bg)")

    model = create_fcos(
        backbone=c["BACKBONE"],
        num_classes=num_classes,
        pretrained=c["PRETRAINED"],
    )
    model = model.to(c["DEVICE"])
    params = count_parameters(model)
    print(f"[FCOS] Total params: {params['total_millions']:.2f}M")

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
        print(f"[FCOS] Resume dir: {save_dir}")

        resume_state = load_resume_state(c["RESUME_FROM"], model, optimizer, c["DEVICE"])
        start_epoch = resume_state["start_epoch"]

        best_state = load_best_state(save_dir / "best.pt")
        best_map50 = best_state["best_map50"]
        best_epoch = best_state["best_epoch"]
        patience_counter = max(0, (start_epoch - 1) - best_epoch)
        print(f"[FCOS] Best so far: mAP@0.5={best_map50:.4f} @ epoch {best_epoch}")
        print(f"[FCOS] Patience counter: {patience_counter}/{c['PATIENCE']}")
    else:
        save_dir = resolve_run_dir(c["PROJECT"], c["NAME"])
        save_dir.mkdir(parents=True, exist_ok=True)
        excel_path = save_dir / c["OUTPUT_XLSX"]
        print(f"[FCOS] Output dir: {save_dir}")
        build_excel(excel_path, class_names)

    val_gt_path = Path(data_dir) / "val" / "_annotations.coco.json"
    coco_gt = COCO(str(val_gt_path))

    print(f"\n[FCOS] Starting training from epoch {start_epoch}...\n")

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
            amp_ctx=amp_ctx,
            scaler=scaler,
            base_lr=lr,
            warmup_iters=c.get("WARMUP_ITERS", 0),
            warmup_factor=c.get("WARMUP_FACTOR", 0.001),
            max_grad_norm=c.get("MAX_GRAD_NORM", None),
        )
        train_total = sum(train_losses.values())
        print(f"  Train losses: bbox={train_losses['bbox_regression']:.4f}  "
              f"cls={train_losses['classification']:.4f}  "
              f"ctr={train_losses['bbox_ctrness']:.4f}  "
              f"total={train_total:.4f}")

        val_results = validate(
            model=model,
            loader=val_loader,
            coco_gt=coco_gt,
            class_names=class_names,
            device=c["DEVICE"],
            amp_ctx=amp_ctx,
        )
        val_losses  = val_results["losses"]
        val_metrics = val_results["metrics"]
        per_class   = val_results["per_class"]
        val_total   = sum(val_losses.values())

        print(f"  Val losses:   bbox={val_losses['bbox_regression']:.4f}  "
              f"cls={val_losses['classification']:.4f}  "
              f"ctr={val_losses['bbox_ctrness']:.4f}  "
              f"total={val_total:.4f}")
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
            print(f"[FCOS] Early stopping at epoch {epoch}. "
                  f"Best: epoch {best_epoch}, mAP@0.5={best_map50:.4f}")
            break

    print(f"\n[FCOS] Training finished.")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Best mAP@0.5: {best_map50:.4f}")
    print(f"  Output dir:   {save_dir}")

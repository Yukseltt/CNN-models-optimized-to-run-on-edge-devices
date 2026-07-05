# distill_yolo.py
#
# Feature-based knowledge distillation for YOLOv8.
# YOLOv8 icin feature-tabanli bilgi damitma.
#
# Teacher (YOLOv8l) transfers neck-feature knowledge to student (YOLOv8n).
# Teacher (YOLOv8l), neck-feature bilgisini student'a (YOLOv8n) aktarir.
#
# Mechanism / Mekanizma:
#     1. Teacher and student both produce P3/P4/P5 neck features (same
#        spatial size, different channel count).
#     1. Teacher ve student ikisi de P3/P4/P5 neck feature uretir (ayni
#        spatial boyut, farkli kanal sayisi).
#     2. A 1x1 conv adapter projects student channels to teacher channels.
#     2. 1x1 conv adapter, student kanallarini teacher kanallarina projekte eder.
#     3. MSE loss between adapted student features and teacher features.
#     3. Adapte edilmis student feature ile teacher feature arasi MSE loss.
#     4. Total loss = Ultralytics detection loss + alpha * feature loss.
#     4. Toplam loss = Ultralytics detection loss + alpha * feature loss.
#
# Implementation note / Implementasyon notu:
#     We bypass Ultralytics' DetectionTrainer entirely and write a plain
#     training loop, reusing only its dataloader builder. This avoids
#     depending on internal trainer callback/scheduler machinery.
#     Ultralytics'in DetectionTrainer'ini tamamen bypass edip duz bir
#     training loop yaziyoruz, sadece dataloader builder'ini tekrar
#     kullaniyoruz. Bu, ic trainer callback/scheduler mekanizmasina
#     bagimliligi onler.

import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.cfg import get_cfg, DEFAULT_CFG


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs_new"


DEFAULT_CFG = {
    "EPOCHS":        50,
    "BATCH_SIZE":    16,
    "PATIENCE":      15,
    "WORKERS":       8,
    "SEED":          0,
    "DEVICE":        "cuda:0",
    "IMGSZ":         640,

    "LR0":           0.001,   # Lower than fresh training: warm-started student.
    "LRF":           0.01,
    "MOMENTUM":      0.937,
    "WEIGHT_DECAY":  0.0005,
    "WARMUP_EPOCHS": 1,

    "TEACHER_PT":    None,    # path to teacher best.pt
    "STUDENT_PT":    None,    # path to student best.pt (warm start)
    "DATA_YAML":     None,

    # Feature distillation weight. Detection loss is the primary signal;
    # this adds the teacher's guidance on top.
    # Feature damitma agirligi. Detection loss ana sinyal; bu, teacher'in
    # rehberligini ustune ekler.
    "ALPHA_FEAT":    1.0,

    "PROJECT":       str(RUNS_DIR),
    "NAME":          "yolov8n_distill_default",
    "LOG_INTERVAL":  50,
}


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


def resolve_run_dir(project_dir: str, base_name: str) -> Path:
    project = Path(project_dir)
    project.mkdir(parents=True, exist_ok=True)
    if not (project / base_name).exists():
        return project / base_name
    counter = 2
    while (project / f"{base_name}{counter}").exists():
        counter += 1
    return project / f"{base_name}{counter}"


# Feature capture via forward hooks / Forward hook ile feature yakalama.

class FeatureCapture:
    # Captures the list of feature maps fed into the Detect head.
    # Detect head'e verilen feature map listesini yakalar.
    def __init__(self, model):
        self.features = None
        detect = model.model[-1]
        self._hook = detect.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        # inputs[0] is the list [P3, P4, P5] fed to Detect.
        # inputs[0], Detect'e verilen [P3, P4, P5] listesidir.
        self.features = list(inputs[0])

    def remove(self):
        self._hook.remove()


class FeatureAdapter(nn.Module):
    # 1x1 conv adapters projecting student channels to teacher channels,
    # one per pyramid level (P3, P4, P5).
    # Student kanallarini teacher kanallarina projekte eden 1x1 conv
    # adapter'lar, her piramit seviyesi (P3, P4, P5) icin bir tane.
    def __init__(self, student_channels: list, teacher_channels: list):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, kernel_size=1)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(self, student_feats: list) -> list:
        return [adapter(f) for adapter, f in zip(self.adapters, student_feats)]


def infer_channels(model, imgsz: int, device: str) -> list:
    # Run a dummy forward pass to discover P3/P4/P5 channel counts.
    # P3/P4/P5 kanal sayilarini bulmak icin dummy forward pass calistirir.
    capture = FeatureCapture(model)
    with torch.no_grad():
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        model(dummy)
    channels = [f.shape[1] for f in capture.features]
    capture.remove()
    return channels


def build_train_loader(data_yaml: str, imgsz: int, batch_size: int, workers: int):
    # Build a training dataloader using Ultralytics' own dataset builder,
    # bypassing DetectionTrainer entirely.
    # DetectionTrainer'i tamamen atlayarak, Ultralytics'in kendi dataset
    # builder'ini kullanarak training dataloader olusturur.
    cfg = get_cfg(overrides={"data": data_yaml, "imgsz": imgsz, "task": "detect"})
    from ultralytics.data.utils import check_det_dataset
    data_info = check_det_dataset(data_yaml)

    dataset = build_yolo_dataset(
        cfg, data_info["train"], batch_size, data_info,
        mode="train", rect=False,
    )
    loader = build_dataloader(
        dataset, batch=batch_size, workers=workers, shuffle=True,
    )
    return loader, data_info


def build_val_loader(data_yaml: str, imgsz: int, batch_size: int, workers: int, data_info: dict):
    cfg = get_cfg(overrides={"data": data_yaml, "imgsz": imgsz, "task": "detect"})
    dataset = build_yolo_dataset(
        cfg, data_info["val"], batch_size, data_info,
        mode="val", rect=False,
    )
    loader = build_dataloader(
        dataset, batch=batch_size, workers=workers, shuffle=False,
    )
    return loader


def preprocess_batch(batch: dict, device: str) -> dict:
    # Move batch tensors to device and normalize images (matches Ultralytics
    # DetectionTrainer.preprocess_batch behavior).
    # Batch tensorlerini device'a tasir ve goruntuleri normalize eder
    # (Ultralytics DetectionTrainer.preprocess_batch davranisiyla esler).
    batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
    for k in ["cls", "bboxes", "batch_idx"]:
        if k in batch:
            batch[k] = batch[k].to(device)
    return batch


def feature_distill_loss(student_feats: list, teacher_feats: list, adapter: FeatureAdapter) -> torch.Tensor:
    # MSE loss between adapted student features and teacher features,
    # averaged across the 3 pyramid levels.
    # Adapte edilmis student feature ile teacher feature arasi MSE loss,
    # 3 piramit seviyesi uzerinden ortalanir.
    adapted = adapter(student_feats)
    losses = []
    for s_adapted, t_feat in zip(adapted, teacher_feats):
        losses.append(F.mse_loss(s_adapted, t_feat.detach()))
    return torch.stack(losses).mean()


@torch.no_grad()
def quick_validate(student_model, val_loader, device: str, max_batches: int = 50) -> float:
    # Lightweight validation loss check (not full mAP) to monitor training
    # health between epochs without the cost of full COCO evaluation.
    # Epoch'lar arasi egitim sagligini izlemek icin hafif validation loss
    # kontrolu (tam mAP degil), tam COCO degerlendirme maliyeti olmadan.
    student_model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = preprocess_batch(batch, device)
        loss, _ = student_model.loss(batch)
        total_loss += loss.sum().item()
        n += 1
    student_model.train()
    return total_loss / max(1, n)


def distill(cfg: Optional[dict] = None) -> None:
    c = {**DEFAULT_CFG, **(cfg or {})}
    set_seed(c["SEED"])

    device = c["DEVICE"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Distill-YOLO] CUDA not available, using CPU.")
        device = "cpu"

    print(f"[Distill-YOLO] Teacher: {c['TEACHER_PT']}")
    print(f"[Distill-YOLO] Student: {c['STUDENT_PT']} (warm start)")
    print(f"[Distill-YOLO] Data:    {c['DATA_YAML']}")
    print(f"[Distill-YOLO] Alpha (feature loss weight): {c['ALPHA_FEAT']}")
    print(f"[Distill-YOLO] Batch={c['BATCH_SIZE']}  Epochs={c['EPOCHS']}  "
          f"LR0={c['LR0']}  Patience={c['PATIENCE']}")

    # Load teacher (frozen).
    # Teacher yukle (donmus).
    teacher_yolo = YOLO(c["TEACHER_PT"])
    teacher = teacher_yolo.model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student (warm start, trainable).
    # Student yukle (warm start, egitilebilir).
    student_yolo = YOLO(c["STUDENT_PT"])
    student = student_yolo.model.to(device)
    student.train()
    # Checkpoint's model.args is a thin dict (task/data/imgsz only), missing
    # the loss-gain hyperparameters (box/cls/dfl) that v8DetectionLoss needs
    # as attribute access. Replace with the full hyperparameter namespace.
    # Checkpoint'in model.args'i ince bir dict (sadece task/data/imgsz),
    # v8DetectionLoss'un attribute olarak ihtiyac duydugu loss-gain
    # hyperparametreleri (box/cls/dfl) eksik. Tam hyperparametre namespace'i
    # ile degistir.
    student.args = DEFAULT_CFG
    student.criterion = student.init_criterion()

    # Discover feature channel counts for adapter construction.
    # Adapter insasi icin feature kanal sayilarini bul.
    print("[Distill-YOLO] Inferring feature channels...")
    teacher.eval()
    t_channels = infer_channels(teacher, c["IMGSZ"], device)
    s_channels = infer_channels(student, c["IMGSZ"], device)
    print(f"  Teacher channels: {t_channels}")
    print(f"  Student channels: {s_channels}")

    adapter = FeatureAdapter(s_channels, t_channels).to(device)

    # Optimizer: student params + adapter params.
    # Optimizer: student parametreleri + adapter parametreleri.
    optimizer = SGD(
        list(student.parameters()) + list(adapter.parameters()),
        lr=c["LR0"], momentum=c["MOMENTUM"], weight_decay=c["WEIGHT_DECAY"],
    )

    # Dataloaders.
    # Dataloader'lar.
    print("[Distill-YOLO] Building dataloaders...")
    train_loader, data_info = build_train_loader(
        c["DATA_YAML"], c["IMGSZ"], c["BATCH_SIZE"], c["WORKERS"],
    )
    val_loader = build_val_loader(
        c["DATA_YAML"], c["IMGSZ"], c["BATCH_SIZE"], c["WORKERS"], data_info,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Output dir.
    # Cikti dizini.
    save_dir = resolve_run_dir(c["PROJECT"], c["NAME"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Distill-YOLO] Output dir: {save_dir}")

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    # Hooks for feature capture, attached once.
    # Feature yakalama icin hook'lar, bir kez takilir.
    t_capture = FeatureCapture(teacher)
    s_capture = FeatureCapture(student)

    print("\n[Distill-YOLO] Starting distillation...\n")

    for epoch in range(c["EPOCHS"]):
        epoch_start = time.time()
        lr = compute_lr(epoch, c["EPOCHS"], c["WARMUP_EPOCHS"], c["LR0"], c["LRF"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        print(f"=== Epoch {epoch}/{c['EPOCHS'] - 1}  lr={lr:.6f} ===")

        student.train()
        sums = {"det": 0.0, "feat": 0.0, "total": 0.0}
        n_batches = 0
        ep_t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = preprocess_batch(batch, device)

            # Teacher forward (no grad), captures features via hook.
            # Teacher forward (no grad), hook ile feature yakalar.
            with torch.no_grad():
                teacher(batch["img"])
            t_feats = [f.clone() for f in t_capture.features]

            # Student forward + detection loss. The forward hook captures
            # student features as a side effect of model.loss -> model(img).
            # Student forward + detection loss. Forward hook, model.loss'un
            # yan etkisi olarak (model.loss -> model(img)) student feature'lari yakalar.
            det_loss, det_loss_items = student.loss(batch)
            det_loss = det_loss.sum()
            s_feats = s_capture.features

            feat_loss = feature_distill_loss(s_feats, t_feats, adapter)

            total_loss = det_loss + c["ALPHA_FEAT"] * feat_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            sums["det"]   += det_loss.item()
            sums["feat"]  += feat_loss.item()
            sums["total"] += total_loss.item()
            n_batches += 1

            if (batch_idx + 1) % c["LOG_INTERVAL"] == 0:
                elapsed = time.time() - ep_t0
                print(f"  [Epoch {epoch}] batch {batch_idx + 1}/{len(train_loader)}  "
                      f"det={sums['det']/n_batches:.4f}  "
                      f"feat={sums['feat']/n_batches:.4f}  "
                      f"total={sums['total']/n_batches:.4f}  "
                      f"elapsed={elapsed:.1f}s")

        avg = {k: v / max(1, n_batches) for k, v in sums.items()}
        print(f"  Train: det={avg['det']:.4f}  feat={avg['feat']:.4f}  total={avg['total']:.4f}")

        # Lightweight validation (loss-based, not full mAP, for speed).
        # Hafif validation (loss-tabanli, tam mAP degil, hiz icin).
        val_loss = quick_validate(student, val_loader, device)
        print(f"  Val (quick, loss-based): {val_loss:.4f}")

        # Save last.
        # last kaydet.
        torch.save(
            {
                "epoch": epoch,
                "model": student.state_dict(),
                "adapter": adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": c,
                "val_loss": val_loss,
            },
            save_dir / "last.pt",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save a YOLO-loadable checkpoint: reuse the student YOLO wrapper's
            # ckpt structure so `YOLO(best.pt)` works downstream.
            # YOLO ile yuklenebilir bir checkpoint kaydet: asagi akiste
            # `YOLO(best.pt)` calissin diye student YOLO wrapper'inin
            # ckpt yapisini tekrar kullan.
            student_yolo.model = student
            student_yolo.save(str(save_dir / "best.pt"))
            print(f"  >> NEW BEST val_loss={best_val_loss:.4f} at epoch {best_epoch}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{c['PATIENCE']})  "
                  f"best={best_val_loss:.4f} @ epoch {best_epoch}")

        print(f"  Epoch time: {time.time() - epoch_start:.1f}s\n")

        if patience_counter >= c["PATIENCE"]:
            print(f"[Distill-YOLO] Early stopping at epoch {epoch}. "
                  f"Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
            break

    t_capture.remove()
    s_capture.remove()

    print(f"\n[Distill-YOLO] Finished. Best epoch {best_epoch}, "
          f"val_loss={best_val_loss:.4f}")
    print(f"  Output: {save_dir}")
    print(f"  NOTE: validate best.pt with the project's mAP benchmark, "
          f"not just this quick loss metric.")
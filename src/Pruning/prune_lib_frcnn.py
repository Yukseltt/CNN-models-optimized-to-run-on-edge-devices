# prune_lib_frcnn.py
#
# torchvision Faster R-CNN (resnet50 / mobilenet) icin BACKBONE-ONLY structured
# pruning + fine-tune pipeline.
#
# NEDEN backbone-only:
#   GeneralizedRCNN.forward trace edilemez (NMS, proposal sampling, dinamik
#   kontrol akisi var) -> torch_pruning DepGraph tum modeli izleyemez. Ama
#   model.backbone (BackboneWithFPN) saf tensor -> feature-map fonksiyonu,
#   trace edilebilir. Backbone'u tek basina budar, FPN cikis kanallarini (256)
#   sabit tutariz -> RPN ve ROI head'ler degismeden calismaya devam eder.
#
# Adimlar:
#   1) BASELINE  : modeli yukle, val mAP olc
#   2) PRUNE     : DepGraph ile backbone ic kanallarini buda (FPN cikislari ignored)
#   3) VERIFY    : pruned backbone ile tam model forward'i dogrula (fail-fast)
#   4) FINE-TUNE : train_faster_rcnn helper'lari ile kucuk lr'de yeniden egit
#   5) FINAL     : eval + .pt (full obj) + (opsiyonel) ONNX
#
# Fine-tune icin train_faster_rcnn.train() KULLANILMAZ: o create_faster_rcnn ile
# modeli yeniden kurar -> pruned yapi kaybolur. Bunun yerine ayni dosyadaki
# yardimci fonksiyonlari (build_dataloaders / train_one_epoch / validate) import
# edip pruned modeli dogrudan egitiriz.

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as nnp        # mask tabanli (sensitivity sweep icin)
import torch_pruning as tp
from torch.optim import SGD
from torch.utils.data import Subset, DataLoader
from torchvision.ops.misc import FrozenBatchNorm2d
from torch_pruning.pruner.function import BasePruningFunc

from prune_common import set_mp_sharing, count_params, reduction_pct, build_budget

set_mp_sharing()


class FrozenBNPruner(BasePruningFunc):
    # torchvision detection backbone'lari standart nn.BatchNorm2d yerine
    # FrozenBatchNorm2d (torchvision.ops.misc) kullanir: weight/bias/running_mean/
    # running_var hepsi BUFFER (parametre degil), num_features attribute yok.
    # torch_pruning bunu tanimaz -> conv cikisi budanir ama ardindaki frozen-BN
    # buffer'lari [C] olarak kalir -> "size of tensor a (44) must match b (64)".
    # Bu pruner dort buffer'i da kanal ekseninde (dim=0) dilimler.
    TARGET_MODULES = FrozenBatchNorm2d

    def prune_out_channels(self, layer, idxs):
        keep = sorted(set(range(layer.weight.shape[0])) - set(idxs))
        keep_t = torch.tensor(keep, dtype=torch.long, device=layer.weight.device)
        layer.weight        = layer.weight.data[keep_t]
        layer.bias          = layer.bias.data[keep_t]
        layer.running_mean  = layer.running_mean.data[keep_t]
        layer.running_var   = layer.running_var.data[keep_t]
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.shape[0]

    def get_in_channels(self, layer):
        return layer.weight.shape[0]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRCNN_DIR = PROJECT_ROOT / "models" / "faster_rcnn"
if str(FRCNN_DIR) not in sys.path:
    sys.path.insert(0, str(FRCNN_DIR))


DEFAULTS = {
    "MODEL_PATH":       None,        # zorunlu: runs/.../best.pt
    "DATA_DIR":         None,        # zorunlu: coco kok dizini (train/ val/)
    "OUT":              None,        # zorunlu: cikti dizini
    "DEVICE":           "cuda:0" if torch.cuda.is_available() else "cpu",
    "TRACE_IMG":        512,         # DepGraph trace icin ornek girdi boyutu

    # Sensitivity sweep: backbone katmanlarina TEK TEK girip mask uygula, mAP
    # dususune bak, per-layer butce cikar (prune_workflow.py'deki YOLO mantigi).
    "DO_SENSITIVITY":   True,        # False -> tum backbone'a duzgun PRUNE_RATIO
    "ORANLAR":          [0.1, 0.2, 0.3, 0.5],   # sweep'te denenen oranlar
    "MIN_FLOOR":        0.10,        # her katman icin garanti min kesim
    "MAX_CEIL":         0.40,        # az hassas katmana izin verilen max kesim
    # Sweep cok yavas (her katman x oran = bir val passi, full val frcnn'de ~74s).
    # Sweep'i val SUBSET'i uzerinde yap (goreli siralama icin yeterli); final
    # eval yine full val. 0/None -> full val (cok uzun surer).
    "SWEEP_VAL_IMAGES": 200,

    # Sweep kapaliyken (DO_SENSITIVITY=False) tum backbone'a uygulanacak duzgun oran.
    "PRUNE_RATIO":      0.30,
    "ROUND_TO":         1,           # kanal yuvarlama (1 = yuvarlama yok, en guvenli)

    # Fine-tune.
    "FINETUNE_EPOCHS":  12,
    "BATCH_SIZE":       8,
    "WORKERS":          4,
    "LR0":              0.001,       # pruned -> baseline'dan kucuk lr
    "LRF":              0.01,
    "MOMENTUM":         0.9,
    "WEIGHT_DECAY":     0.0005,
    "WARMUP_EPOCHS":    1,
    "LOG_INTERVAL":     50,

    # Akis kontrolu.
    "DO_FINETUNE":      True,
    "SKIP_BASELINE":    False,       # True ise baseline val atlanir (hizli smoke)
    "DO_ONNX":          False,       # frcnn ONNX export kirilgan; varsayilan kapali
}


def _prunable_backbone_convs(backbone):
    # Backbone'da prunable conv'lar: FPN degil, 1x1 degil, depthwise degil.
    out = []
    for name, m in backbone.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if "fpn" in name:
            continue
        if m.kernel_size == (1, 1):
            continue
        if m.groups == m.in_channels and m.groups > 1:
            continue  # depthwise -> atla
        out.append((name, m))
    return out


def _prune_backbone(model, trace_img, device, round_to,
                    uniform_ratio=None, ratio_dict_names=None):
    # model.backbone'u tek basina budar (FPN cikislari ignored -> 256 sabit).
    #   uniform_ratio verilirse tum prunable conv'lara o oran,
    #   ratio_dict_names ({layer_name: oran}) verilirse per-layer oran uygulanir.
    backbone = model.backbone

    # KRITIK: inference ckpt parametreleri DONMUS yukleyebilir; DepGraph autograd
    # grad_fn uzerinden kurulur, grad akmazsa graf baglanmaz. Trace oncesi ac.
    for p in backbone.parameters():
        p.requires_grad_(True)

    # Backbone OrderedDict doner; DepGraph icin liste dondurelim.
    class _BBWrap(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb

        def forward(self, x):
            out = self.bb(x)
            return list(out.values())

    wrapper = _BBWrap(backbone).to(device).eval()
    example = torch.randn(1, 3, trace_img, trace_img).to(device)

    # FPN conv'larini ignored yap -> cikis kanali 256 sabit kalir, RPN/ROI bozulmaz.
    # (FPN inner_blocks in_channels backbone govdesi budanirsa otomatik kucukulur.)
    ignored = [m for n, m in backbone.named_modules()
               if isinstance(m, nn.Conv2d) and "fpn" in n]

    shape_before = {n: m.weight.shape for n, m in backbone.named_modules()
                    if isinstance(m, nn.Conv2d)}

    # Per-layer butce -> module->oran sozlugu.
    ratio_dict = None
    if ratio_dict_names is not None:
        name2mod = dict(backbone.named_modules())
        ratio_dict = {name2mod[n]: r for n, r in ratio_dict_names.items()
                      if n in name2mod and r > 0.0}

    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        wrapper,
        example_inputs=example,
        importance=importance,
        pruning_ratio=(0.0 if ratio_dict else uniform_ratio),
        pruning_ratio_dict=ratio_dict,
        ignored_layers=ignored,
        global_pruning=False,
        round_to=round_to,
        # FrozenBatchNorm2d'i tani -> conv ile birlikte kanal eksinde dilimle.
        customized_pruners={FrozenBatchNorm2d: FrozenBNPruner()},
    )
    pruner.step()

    for m in backbone.modules():
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
    backbone.zero_grad(set_to_none=True)

    degisen = sum(1 for n, m in backbone.named_modules()
                  if isinstance(m, nn.Conv2d) and n in shape_before
                  and m.weight.shape != shape_before[n])
    print("backbone shape degisen conv sayisi:", degisen)
    if degisen == 0:
        raise RuntimeError("Hicbir backbone conv budanmadi -> kontrol et "
                           "(requires_grad / ignored_layers).")
    return model


def _quick_map50(model, loader, coco_gt, device):
    # Hizli mAP@0.5: tek pass tahmin + COCOeval. KRITIK: params.imgIds'i SADECE
    # bu loader'in (subset) goruntulerine kisitla -> yoksa COCOeval tum val GT'si
    # uzerinden olcer, subset tahminleri ~100x deflate olur ve sensitivity
    # sinyali kaybolur. summarize() ciktisi bastirilir (228x12 satir gurultu olmasin).
    import io
    import contextlib
    from pycocotools.cocoeval import COCOeval
    from train_faster_rcnn import predictions_to_coco_format

    model.eval()
    results, img_ids = [], set()
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            preds = model(images)
            ids = [int(t["image_id"].item()) for t in targets]
            img_ids.update(ids)
            results.extend(predictions_to_coco_format(preds, ids))
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ce.params.imgIds = sorted(img_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ce.evaluate(); ce.accumulate(); ce.summarize()
    return float(ce.stats[1])   # mAP@0.5


def _sweep_frcnn(model, prunable, oranlar, sweep_loader, coco_gt, device):
    # Backbone katmanlarina TEK TEK girip mask uygula -> subset mAP olc ->
    # orijinal agirligi geri yukle. (prune_workflow.py'deki YOLO sweep'i.)
    base = _quick_map50(model, sweep_loader, coco_gt, device)
    print(f"[sweep] subset baseline mAP@0.5={base:.4f}  ({len(prunable)} katman)")
    sens = {}
    for i, (name, layer) in enumerate(prunable):
        sens[name] = {}
        orig = layer.weight.detach().clone()
        for oran in oranlar:
            # L2 norm en dusuk %oran cikis kanalini sifirla (dim=0), agirliga bak.
            nnp.ln_structured(layer, name="weight", amount=oran, n=2, dim=0)
            nnp.remove(layer, "weight")
            m = _quick_map50(model, sweep_loader, coco_gt, device)
            layer.weight.data.copy_(orig)   # orijinali geri yukle
            sens[name][oran] = base - m
            print(f"[sweep {i+1}/{len(prunable)}] {name} oran={oran} "
                  f"mAP={m:.4f} dusus={base - m:.4f}")
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return sens


def prune_frcnn(cfg: dict) -> dict:
    c = {**DEFAULTS, **cfg}
    for req in ("MODEL_PATH", "DATA_DIR", "OUT"):
        if not c[req]:
            raise ValueError(f"cfg['{req}'] zorunlu.")

    DEVICE = c["DEVICE"]
    OUT = Path(c["OUT"]); OUT.mkdir(parents=True, exist_ok=True)

    from model_loaders import load_faster_rcnn
    from train_faster_rcnn import (
        build_dataloaders, train_one_epoch, validate, compute_lr, set_seed,
    )
    from dataset import collate_fn
    from pycocotools.coco import COCO

    # =====================================================================
    # 1) BASELINE
    # =====================================================================
    print("=" * 60); print("1) BASELINE"); print("=" * 60)
    model, class_names, frcnn_cfg = load_faster_rcnn(c["MODEL_PATH"], DEVICE)
    backbone_name = frcnn_cfg.get("BACKBONE", "resnet50")
    print(f"[FRCNN] backbone={backbone_name}, classes={class_names}")

    baseline_real = count_params(model)
    print("params (real) =", baseline_real / 1e6, "M")

    # Dataloader'lar (fine-tune + val icin).
    dl_cfg = {"BATCH_SIZE": c["BATCH_SIZE"], "WORKERS": c["WORKERS"]}
    train_loader, val_loader, ds_class_names, num_classes = build_dataloaders(
        c["DATA_DIR"], dl_cfg
    )
    val_gt = COCO(str(Path(c["DATA_DIR"]) / "val" / "_annotations.coco.json"))

    baseline_map = None
    if not c["SKIP_BASELINE"]:
        res = validate(model, val_loader, val_gt, ds_class_names, DEVICE)
        baseline_map = res["metrics"]["map50"]
        print(f"baseline mAP@0.5={baseline_map:.4f}  "
              f"mAP@0.5:0.95={res['metrics']['map50_95']:.4f}")

    # =====================================================================
    # 2) PRUNE (backbone-only)  — sweep'li per-layer butce veya duzgun oran
    # =====================================================================
    print("=" * 60); print("2) PRUNE (backbone-only, DepGraph)"); print("=" * 60)
    butce_names = None
    if c["DO_SENSITIVITY"]:
        prunable = _prunable_backbone_convs(model.backbone)
        print(f"[FRCNN] prunable backbone conv: {len(prunable)} katman")

        # Sweep'i hizlandirmak icin val SUBSET'i (goreli siralama icin yeterli).
        n_sub = c["SWEEP_VAL_IMAGES"]
        if n_sub and n_sub < len(val_loader.dataset):
            sub_idx = list(range(n_sub))
            sweep_loader = DataLoader(
                Subset(val_loader.dataset, sub_idx),
                batch_size=c["BATCH_SIZE"], shuffle=False, num_workers=c["WORKERS"],
                collate_fn=collate_fn, pin_memory=True,
            )
            print(f"[sweep] val subset = ilk {n_sub} goruntu")
        else:
            sweep_loader = val_loader
            print("[sweep] full val (subset kapali) - UZUN surebilir")

        sensitivity = _sweep_frcnn(model, prunable, c["ORANLAR"], sweep_loader,
                                   val_gt, DEVICE)
        butce_names, skor = build_budget(
            sensitivity, [n for n, _ in prunable], c["MIN_FLOOR"], c["MAX_CEIL"],
        )
        with open(OUT / "sensitivity.json", "w") as f:
            json.dump(sensitivity, f, indent=2)
        with open(OUT / "butce.json", "w") as f:
            json.dump(butce_names, f, indent=2)
        print("[FRCNN] per-layer butce:",
              {k: round(v, 2) for k, v in list(butce_names.items())[:6]}, "...")
        model = _prune_backbone(model, c["TRACE_IMG"], DEVICE, c["ROUND_TO"],
                                ratio_dict_names=butce_names)
    else:
        print(f"[FRCNN] sweep kapali -> tum backbone'a duzgun oran={c['PRUNE_RATIO']}")
        model = _prune_backbone(model, c["TRACE_IMG"], DEVICE, c["ROUND_TO"],
                                uniform_ratio=c["PRUNE_RATIO"])
    yeni_real = count_params(model)
    print("KESIM SONRASI params (real) =", yeni_real / 1e6, "M",
          " azalma=", reduction_pct(baseline_real, yeni_real), "%")

    # =====================================================================
    # 3) VERIFY (pruned backbone -> tam model forward calisiyor mu?)
    # =====================================================================
    print("=" * 60); print("3) VERIFY tam model forward"); print("=" * 60)
    model.eval().to(DEVICE)
    with torch.no_grad():
        dummy = [torch.randn(3, c["TRACE_IMG"], c["TRACE_IMG"]).to(DEVICE)]
        out = model(dummy)
    print("forward OK, ornek cikti anahtarlari:", list(out[0].keys()))

    # =====================================================================
    # 4) FINE-TUNE (pruned modeli koruyarak)
    # =====================================================================
    if c["DO_FINETUNE"]:
        print("=" * 60); print("4) FINE-TUNE"); print("=" * 60)
        set_seed(0)
        for p in model.parameters():
            p.requires_grad_(True)
        optimizer = SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=c["LR0"], momentum=c["MOMENTUM"], weight_decay=c["WEIGHT_DECAY"],
        )

        best_map = -1.0
        for epoch in range(c["FINETUNE_EPOCHS"]):
            lr = compute_lr(epoch, c["FINETUNE_EPOCHS"], c["WARMUP_EPOCHS"],
                            c["LR0"], c["LRF"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            t0 = time.time()
            print(f"=== Epoch {epoch}/{c['FINETUNE_EPOCHS'] - 1}  lr={lr:.6f} ===")
            tl = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch,
                                 log_interval=c["LOG_INTERVAL"])
            print(f"  train total loss={sum(tl.values()):.4f}")
            res = validate(model, val_loader, val_gt, ds_class_names, DEVICE)
            vm = res["metrics"]
            print(f"  val mAP@0.5={vm['map50']:.4f}  "
                  f"mAP@0.5:0.95={vm['map50_95']:.4f}  ({time.time()-t0:.0f}s)")

            torch.save(
                {"model_obj": model, "state_dict": model.state_dict(),
                 "class_names": class_names, "cfg": c, "epoch": epoch,
                 "metric": vm["map50"], "backbone": backbone_name},
                OUT / "last.pt",
            )
            if vm["map50"] > best_map:
                best_map = vm["map50"]
                torch.save(
                    {"model_obj": model, "state_dict": model.state_dict(),
                     "class_names": class_names, "cfg": c, "epoch": epoch,
                     "metric": vm["map50"], "backbone": backbone_name},
                    OUT / "best.pt",
                )
                print(f"  >> NEW BEST mAP@0.5={best_map:.4f}")
        final_map = best_map
    else:
        print("[FRCNN] fine-tune atlandi (DO_FINETUNE=False).")
        # Fine-tune'suz da KESIM SONRASI mAP olc -> budamanin dogruluga
        # zararini (fine-tune oncesi) gormek icin.
        final_map = None
        if not c["SKIP_BASELINE"]:
            print("Kesim sonrasi eval (fine-tune YOK)...")
            res = validate(model, val_loader, val_gt, ds_class_names, DEVICE)
            final_map = res["metrics"]["map50"]
            print(f"  kesim sonrasi mAP@0.5={final_map:.4f}  "
                  f"mAP@0.5:0.95={res['metrics']['map50_95']:.4f}")
        torch.save(
            {"model_obj": model, "state_dict": model.state_dict(),
             "class_names": class_names, "cfg": c, "epoch": -1,
             "metric": final_map, "backbone": backbone_name},
            OUT / "pruned_final.pt",
        )

    # =====================================================================
    # 5) FINAL + (opsiyonel) ONNX
    # =====================================================================
    print("=" * 60); print("5) FINAL"); print("=" * 60)
    print("mAP@0.5:", baseline_map, "->", final_map)
    print("params (real):", baseline_real / 1e6, "M ->", yeni_real / 1e6, "M",
          " azalma=", reduction_pct(baseline_real, yeni_real), "%")

    if c["DO_ONNX"]:
        # Faster R-CNN ONNX export kirilgan (dinamik output). Best-effort.
        try:
            onnx_yol = OUT / "pruned_final.onnx"
            model.eval()
            torch.onnx.export(
                model, [torch.randn(3, c["TRACE_IMG"], c["TRACE_IMG"]).to(DEVICE)],
                str(onnx_yol), opset_version=17,
                input_names=["images"],
            )
            print("ONNX:", onnx_yol)
        except Exception as e:
            print("ONNX export atlandi (frcnn dinamik output):", e)

    return {
        "baseline_map": baseline_map,
        "final_map": final_map,
        "params_before_M": baseline_real / 1e6,
        "params_after_M": yeni_real / 1e6,
        "reduction_pct": reduction_pct(baseline_real, yeni_real),
        "out": str(OUT),
    }

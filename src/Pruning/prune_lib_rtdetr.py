# prune_lib_rtdetr.py
#
# HF RT-DETR v2 (PekingU/rtdetr_v2_r18vd) icin BACKBONE-ONLY structured pruning
# + HF Trainer ile fine-tune pipeline.
#
# NEDEN backbone-only:
#   Transformer encoder/decoder'in (attention head dim, layernorm, deformable
#   sampling) structured pruning'i kirilgan. Onun yerine sadece ResNet18
#   backbone'unu budar, transformer'i (d_model=256) DEGISMEDEN tutariz.
#
# NASIL d_model sabit kaliyor:
#   Backbone'un cikis feature-map'leri encoder_input_proj (3 adet 1x1 conv,
#   [128/256/512] -> 256) uzerinden transformer'a girer. DepGraph'i sadece
#   (inner_resnet + encoder_input_proj) sarmalayan kucuk bir wrapper uzerinde
#   kurar, encoder_input_proj conv'larini ignored yapariz -> cikis 256 sabit
#   kalir, backbone ic+cikis kanallari budanir, proj in_channels otomatik kucukulur.
#
# Adimlar:
#   1) LOAD      : best.pt state_dict'inden HF modeli kur
#   2) BASELINE  : Trainer.evaluate() ile val mAP (opsiyonel)
#   3) PRUNE     : DepGraph ile backbone'u buda
#   4) VERIFY    : tam model forward'i dogrula (fail-fast)
#   5) FINE-TUNE : HF Trainer ile pruned modeli yeniden egit
#   6) FINAL     : eval + .pt (full obj)

import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as nnp        # mask tabanli (sensitivity sweep icin)
import torch_pruning as tp
from torch.utils.data import Subset
from torch_pruning.pruner.function import BasePruningFunc
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2FrozenBatchNorm2d,
)

from prune_common import set_mp_sharing, count_params, reduction_pct, build_budget

set_mp_sharing()


class FrozenBNPruner(BasePruningFunc):
    # RT-DETR v2'nin ResNet backbone'u RTDetrV2FrozenBatchNorm2d kullanir:
    # weight/bias/running_mean/running_var hepsi BUFFER (parametre degil).
    # torch_pruning bunu tanimaz -> conv cikisi budanir ama frozen-BN buffer'lari
    # [C] kalir -> "size of tensor a (22) must match b (32)". Bu pruner dort
    # buffer'i da kanal eksinde (dim=0) dilimler.
    TARGET_MODULES = RTDetrV2FrozenBatchNorm2d

    def prune_out_channels(self, layer, idxs):
        keep = sorted(set(range(layer.weight.shape[0])) - set(idxs))
        keep_t = torch.tensor(keep, dtype=torch.long, device=layer.weight.device)
        layer.weight       = layer.weight.data[keep_t]
        layer.bias         = layer.bias.data[keep_t]
        layer.running_mean = layer.running_mean.data[keep_t]
        layer.running_var  = layer.running_var.data[keep_t]
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.shape[0]

    def get_in_channels(self, layer):
        return layer.weight.shape[0]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTDETR_DIR = PROJECT_ROOT / "models" / "rtdetr_v2_r18vd"
if str(RTDETR_DIR) not in sys.path:
    sys.path.insert(0, str(RTDETR_DIR))


DEFAULTS = {
    "MODEL_PATH":       None,        # zorunlu: runs/.../best.pt
    "DATA_DIR":         None,        # zorunlu: coco kok dizini (train/ val/)
    "OUT":              None,        # zorunlu: cikti dizini
    "CHECKPOINT":       "PekingU/rtdetr_v2_r18vd",  # config + image_processor kaynagi
    "IMAGE_SIZE":       480,
    "DEVICE":           "cuda:0" if torch.cuda.is_available() else "cpu",

    # Sensitivity sweep: backbone (ResNet18) katmanlarina TEK TEK girip mask
    # uygula, mAP dususune bak, per-layer butce cikar (YOLO prune_workflow mantigi).
    "DO_SENSITIVITY":   True,        # False -> tum backbone'a duzgun PRUNE_RATIO
    "ORANLAR":          [0.1, 0.2, 0.3, 0.5],
    "MIN_FLOOR":        0.10,
    "MAX_CEIL":         0.40,
    # Sweep'i val SUBSET'i uzerinde yap (goreli siralama icin yeterli). torchmetrics
    # sadece update'lenen goruntuler uzerinden olcer -> subset deflate sorunu YOK.
    "SWEEP_VAL_IMAGES": 200,

    # Sweep kapaliyken (DO_SENSITIVITY=False) tum backbone'a uygulanacak duzgun oran.
    "PRUNE_RATIO":      0.30,
    "ROUND_TO":         1,

    # Fine-tune.
    "FINETUNE_EPOCHS":  15,
    "TRAIN_BS":         16,
    "EVAL_BS":          32,
    "LR":               2e-5,        # pruned -> kucuk lr
    "WARMUP_STEPS":     100,
    "WEIGHT_DECAY":     1e-4,
    "MAX_GRAD_NORM":    0.1,
    "WORKERS":          6,
    "SEED":             42,

    # Akis kontrolu.
    "DO_FINETUNE":      True,
    "SKIP_BASELINE":    False,
}


def _load_model(model_path, checkpoint, device):
    from transformers import AutoConfig, AutoModelForObjectDetection
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}

    config = AutoConfig.from_pretrained(checkpoint)
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(class_names)
    model = AutoModelForObjectDetection.from_config(config)

    sd = ckpt["model"]
    # torch.compile ile kaydedilmis olabilir -> "_orig_mod." prefix temizle.
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[RT-DETR] load_state_dict missing={len(missing)} (orn: {missing[:2]})")
    if unexpected:
        print(f"[RT-DETR] load_state_dict unexpected={len(unexpected)} (orn: {unexpected[:2]})")
    model.to(device)
    return model, class_names, id2label


def _prunable_backbone_convs(inner):
    # Inner ResNet'te prunable conv'lar: 1x1 degil, depthwise degil.
    out = []
    for name, m in inner.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.kernel_size == (1, 1):
            continue
        if m.groups == m.in_channels and m.groups > 1:
            continue
        out.append((name, m))
    return out


def _prune_backbone(model, image_size, device, round_to,
                    uniform_ratio=None, ratio_dict_names=None):
    inner = model.model.backbone.model        # RTDetrResNetBackbone
    proj  = model.model.encoder_input_proj     # ModuleList[Sequential(Conv2d, BN)]

    for p in inner.parameters():
        p.requires_grad_(True)
    for p in proj.parameters():
        p.requires_grad_(True)

    class _BBProjWrap(nn.Module):
        # Backbone + encoder_input_proj'u izlenebilir tek bir tensor->liste
        # fonksiyonu olarak sarmalar. proj conv'lari ignored -> cikis 256 sabit.
        def __init__(self, bb, proj):
            super().__init__()
            self.bb = bb
            self.proj = proj

        def forward(self, x):
            fm = self.bb(x).feature_maps
            return [p(f) for p, f in zip(self.proj, fm)]

    wrapper = _BBProjWrap(inner, proj).to(device).eval()
    example = torch.randn(1, 3, image_size, image_size).to(device)

    # encoder_input_proj icindeki Conv2d'leri ignored yap -> out=256 (d_model) sabit.
    ignored = [m for m in proj.modules() if isinstance(m, nn.Conv2d)]

    shape_before = {n: m.weight.shape for n, m in inner.named_modules()
                    if isinstance(m, nn.Conv2d)}

    # Per-layer butce -> module->oran sozlugu.
    ratio_dict = None
    if ratio_dict_names is not None:
        name2mod = dict(inner.named_modules())
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
        # RTDetrV2FrozenBatchNorm2d'i tani -> conv ile birlikte kanal eksinde dilimle.
        customized_pruners={RTDetrV2FrozenBatchNorm2d: FrozenBNPruner()},
    )
    pruner.step()

    for m in inner.modules():
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
    inner.zero_grad(set_to_none=True)

    degisen = sum(1 for n, m in inner.named_modules()
                  if isinstance(m, nn.Conv2d) and n in shape_before
                  and m.weight.shape != shape_before[n])
    print("backbone shape degisen conv sayisi:", degisen)
    if degisen == 0:
        raise RuntimeError("Hicbir backbone conv budanmadi -> kontrol et "
                           "(requires_grad / ignored_layers).")
    return model


def _sweep_rtdetr(prunable, oranlar, eval_fn):
    # Inner ResNet katmanlarina TEK TEK girip mask uygula -> subset eval_map_50
    # olc -> orijinal agirligi geri yukle. eval_fn() float map50 doner.
    base = eval_fn()
    print(f"[sweep] subset baseline eval_map_50={base:.4f}  ({len(prunable)} katman)")
    sens = {}
    for i, (name, layer) in enumerate(prunable):
        sens[name] = {}
        orig = layer.weight.detach().clone()
        for oran in oranlar:
            nnp.ln_structured(layer, name="weight", amount=oran, n=2, dim=0)
            nnp.remove(layer, "weight")
            m = eval_fn()
            layer.weight.data.copy_(orig)
            sens[name][oran] = base - m
            print(f"[sweep {i+1}/{len(prunable)}] {name} oran={oran} "
                  f"map50={m:.4f} dusus={base - m:.4f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return sens


def prune_rtdetr(cfg: dict) -> dict:
    c = {**DEFAULTS, **cfg}
    for req in ("MODEL_PATH", "DATA_DIR", "OUT"):
        if not c[req]:
            raise ValueError(f"cfg['{req}'] zorunlu.")

    DEVICE = c["DEVICE"]
    OUT = Path(c["OUT"]); OUT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(c["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["SEED"])
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    from transformers import (
        AutoImageProcessor, Trainer, TrainingArguments,
    )
    from dataset import (
        LocalCocoDataset, ThermalCocoDataset, build_transforms, collate_fn,
    )
    from map_evaluator import MAPEvaluator

    # =====================================================================
    # 1) LOAD
    # =====================================================================
    print("=" * 60); print("1) LOAD"); print("=" * 60)
    model, class_names, id2label = _load_model(c["MODEL_PATH"], c["CHECKPOINT"], DEVICE)
    baseline_real = count_params(model)
    print(f"[RT-DETR] classes={class_names}  params={baseline_real/1e6:.2f}M")

    image_processor = AutoImageProcessor.from_pretrained(
        c["CHECKPOINT"], do_resize=True,
        size={"width": c["IMAGE_SIZE"], "height": c["IMAGE_SIZE"]}, use_fast=True,
    )
    train_aug, val_aug = build_transforms()
    train_base = LocalCocoDataset(Path(c["DATA_DIR"]) / "train")
    val_base   = LocalCocoDataset(Path(c["DATA_DIR"]) / "val")
    train_ds = ThermalCocoDataset(train_base, image_processor, transform=train_aug)
    val_ds   = ThermalCocoDataset(val_base,   image_processor, transform=val_aug)
    print(f"[RT-DETR] Train {len(train_base)} / Val {len(val_base)} imgs")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    # TF32 sadece Ampere+ (compute capability >= 8.0) GPU'larda mevcut (T4 = 7.5).
    use_tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    args = TrainingArguments(
        output_dir                  = str(OUT / "hf_checkpoints"),
        num_train_epochs            = c["FINETUNE_EPOCHS"],
        learning_rate               = c["LR"],
        warmup_steps                = c["WARMUP_STEPS"],
        weight_decay                = c["WEIGHT_DECAY"],
        max_grad_norm               = c["MAX_GRAD_NORM"],
        per_device_train_batch_size = c["TRAIN_BS"],
        per_device_eval_batch_size  = c["EVAL_BS"],
        dataloader_num_workers      = c["WORKERS"],
        dataloader_pin_memory       = True,
        dataloader_persistent_workers = c["WORKERS"] > 0,
        dataloader_prefetch_factor  = 2 if c["WORKERS"] > 0 else None,
        dataloader_drop_last        = True,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        # KRITIK: bf16_full_eval=True, evaluate sonrasi model PARAMLARINI kalici
        # bf16'ya cevirir -> sonraki prune/verify float32 input ile dtype catismasi
        # ("Input FloatTensor vs weight BFloat16"). False -> eval autocast ile
        # bf16 hesaplar ama paramlar fp32 kalir; pruning/verify tutarli.
        bf16_full_eval              = False,
        tf32                        = use_tf32,
        optim                       = "adamw_torch_fused",
        eval_accumulation_steps     = 4,
        eval_strategy               = c.get("EVAL_STRATEGY", "no"),  # "epoch" -> her epoch eval (ilerleme gormek icin)
        save_strategy               = "no",     # pruned mimari from_pretrained ile geri yuklenemez
        logging_steps               = 50,
        remove_unused_columns       = False,
        eval_do_concat_batches      = False,
        report_to                   = "none",
        push_to_hub                 = False,
        seed                        = c["SEED"],
    )

    eval_fn = MAPEvaluator(image_processor=image_processor, threshold=0.0,
                           id2label=id2label)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=image_processor, data_collator=collate_fn,
        compute_metrics=eval_fn,
    )

    # =====================================================================
    # 2) BASELINE
    # =====================================================================
    baseline_map = None
    if not c["SKIP_BASELINE"]:
        print("=" * 60); print("2) BASELINE eval"); print("=" * 60)
        m = trainer.evaluate()
        baseline_map = float(m.get("eval_map_50", m.get("eval_map", 0.0)))
        print(f"baseline eval_map={m.get('eval_map')}  eval_map_50={m.get('eval_map_50')}")

    # =====================================================================
    # 3) PRUNE (backbone-only)  — sweep'li per-layer butce veya duzgun oran
    # =====================================================================
    print("=" * 60); print("3) PRUNE (backbone-only, DepGraph)"); print("=" * 60)
    model.float()   # defansif: evaluate sonrasi olasi bf16 cast'i geri al (DepGraph fp32 ister)
    if c["DO_SENSITIVITY"]:
        inner = model.model.backbone.model
        prunable = _prunable_backbone_convs(inner)
        print(f"[RT-DETR] prunable backbone conv: {len(prunable)} katman")

        # Sweep'i val SUBSET'i uzerinde yap (torchmetrics subset deflate etmez).
        n_sub = c["SWEEP_VAL_IMAGES"]
        sweep_ds = val_ds
        if n_sub and n_sub < len(val_ds):
            sweep_ds = Subset(val_ds, list(range(n_sub)))
            print(f"[sweep] val subset = ilk {n_sub} goruntu")
        else:
            print("[sweep] full val (subset kapali) - UZUN surebilir")

        def _eval_subset():
            mm = trainer.evaluate(eval_dataset=sweep_ds)
            return float(mm.get("eval_map_50", mm.get("eval_map", 0.0)))

        sensitivity = _sweep_rtdetr(prunable, c["ORANLAR"], _eval_subset)
        butce_names, skor = build_budget(
            sensitivity, [n for n, _ in prunable], c["MIN_FLOOR"], c["MAX_CEIL"],
        )
        with open(OUT / "sensitivity.json", "w") as f:
            json.dump(sensitivity, f, indent=2)
        with open(OUT / "butce.json", "w") as f:
            json.dump(butce_names, f, indent=2)
        print("[RT-DETR] per-layer butce:",
              {k: round(v, 2) for k, v in list(butce_names.items())[:6]}, "...")
        model = _prune_backbone(model, c["IMAGE_SIZE"], DEVICE, c["ROUND_TO"],
                                ratio_dict_names=butce_names)
    else:
        print(f"[RT-DETR] sweep kapali -> tum backbone'a duzgun oran={c['PRUNE_RATIO']}")
        model = _prune_backbone(model, c["IMAGE_SIZE"], DEVICE, c["ROUND_TO"],
                                uniform_ratio=c["PRUNE_RATIO"])
    yeni_real = count_params(model)
    print("KESIM SONRASI params (real) =", yeni_real / 1e6, "M",
          " azalma=", reduction_pct(baseline_real, yeni_real), "%")
    trainer.model = model  # pruned modeli Trainer'a geri ver

    # =====================================================================
    # 4) VERIFY
    # =====================================================================
    print("=" * 60); print("4) VERIFY tam model forward"); print("=" * 60)
    model.eval().to(DEVICE)
    with torch.no_grad():
        out = model(pixel_values=torch.randn(1, 3, c["IMAGE_SIZE"], c["IMAGE_SIZE"]).to(DEVICE))
    print("forward OK, logits shape:", tuple(out.logits.shape))

    # =====================================================================
    # 5) FINE-TUNE
    # =====================================================================
    final_map = None
    if c["DO_FINETUNE"]:
        print("=" * 60); print("5) FINE-TUNE (HF Trainer)"); print("=" * 60)
        trainer.train()
        m = trainer.evaluate()
        final_map = float(m.get("eval_map_50", m.get("eval_map", 0.0)))
        print(f"final eval_map={m.get('eval_map')}  eval_map_50={m.get('eval_map_50')}")
    else:
        print("[RT-DETR] fine-tune atlandi (DO_FINETUNE=False).")
        # Fine-tune'suz da KESIM SONRASI mAP olc -> budamanin dogruluga
        # zararini (fine-tune oncesi) gormek icin.
        if not c["SKIP_BASELINE"]:
            print("Kesim sonrasi eval (fine-tune YOK)...")
            m = trainer.evaluate()
            final_map = float(m.get("eval_map_50", m.get("eval_map", 0.0)))
            print(f"kesim sonrasi eval_map={m.get('eval_map')}  "
                  f"eval_map_50={m.get('eval_map_50')}")

    # =====================================================================
    # 6) FINAL + SAVE
    # =====================================================================
    print("=" * 60); print("6) FINAL"); print("=" * 60)
    print("eval_map_50:", baseline_map, "->", final_map)
    print("params (real):", baseline_real / 1e6, "M ->", yeni_real / 1e6, "M",
          " azalma=", reduction_pct(baseline_real, yeni_real), "%")

    # Pruned mimari config'ten rebuild edilemez -> full obj + state_dict birlikte.
    pt_yol = OUT / ("best.pt" if c["DO_FINETUNE"] else "pruned_final.pt")
    torch.save(
        {"model_obj": model, "state_dict": model.state_dict(),
         "class_names": class_names, "cfg": c, "metric": final_map},
        str(pt_yol),
    )
    print("kaydedildi:", pt_yol)

    return {
        "baseline_map": baseline_map,
        "final_map": final_map,
        "params_before_M": baseline_real / 1e6,
        "params_after_M": yeni_real / 1e6,
        "reduction_pct": reduction_pct(baseline_real, yeni_real),
        "out": str(OUT),
    }

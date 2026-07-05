# prune_lib_yolo.py
#
# Ultralytics YOLO modelleri icin structured pruning pipeline.
# prune_workflow.py'nin (tek YOLO icin yazilmis) genellestirilmis, fonksiyon
# halindeki versiyonu. yolo_fir_v5s ve yolov8n gibi farkli YOLO'lar icin
# tek tek entry script'ten cagrilir.
#
# Adimlar:
#   1) BASELINE  : mAP + param/MAC olc
#   2) SENSITIVITY SWEEP : her katmani sirayla mask'le, mAP dususune bak
#   3) BUTCE     : sensitivity'ye gore per-layer kesim orani (MIN_FLOOR..MAX_CEIL)
#   4) DEPGRAPH  : torch_pruning ile fiziksel kanal kesimi
#   5) FINE-TUNE : pruned modeli koruyarak kucuk lr ile yeniden egit
#   6) FINAL     : eval + .pt + ONNX export
#
# Detect head ve ilk conv (model.0) otomatik tespit edilip SKIP edilir;
# boylece v5su (Detect=model.24) ve v8n (Detect=model.22) ayni kodla calisir.
#
# Kutuphane: torch_pruning (DepGraph) - kanali fiziksel kesiyor.

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as nnp   # mask tabanli (sensitivity icin)
import torch_pruning as tp           # gercek kesim icin
from ultralytics import YOLO

from prune_common import set_mp_sharing, count_params, reduction_pct, slim_onnx

set_mp_sharing()


# Varsayilan ayarlar. Entry script bunlari cfg ile ezer.
DEFAULTS = {
    "MODEL_PATH":       None,        # zorunlu: best.pt yolu
    "DATA_YAML":        None,        # zorunlu: data.yaml yolu
    "OUT":              None,        # zorunlu: cikti dizini
    "IMG":              640,
    "DEVICE":           "cuda" if torch.cuda.is_available() else "cpu",

    # Sensitivity sweep'te denenen oranlar.
    "ORANLAR":          [0.1, 0.2, 0.3, 0.5],
    # Her prunable katman icin GARANTI minimum kesim orani.
    # DIKKAT: 0.0 + "drop <= tolerans" mantigi = HICBIR sey kesilmez; structured
    # pruning fine-tune'dan ONCE mAP'i her zaman dusurur. Tabani >0 tut.
    "MIN_FLOOR":        0.10,
    # Az hassas katmana izin verilen en yuksek oran (MIN_FLOOR..MAX_CEIL).
    "MAX_CEIL":         0.40,
    # Kanal yuvarlama: 2 -> kanal sayilari cift kalir (C2f split((c,c)) icin onemli).
    "ROUND_TO":         2,

    # Fine-tune.
    "FINETUNE_EPOCHS":  10,
    "BATCH":            8,
    "WORKERS":          4,
    "LR0":              0.001,

    # Akis kontrolu (smoke test icin kapatilabilir).
    "DO_SENSITIVITY":   True,        # False ise tum katmanlara MIN_FLOOR uygulanir
    "DO_FINETUNE":      True,
    "SKIP_BASELINE":    False,       # True ise baseline val atlanir (hizli smoke)
    "DO_ONNX":          True,

    # TEST_MODE: kisa duman testi (az katman + tek oran).
    "TEST_MODE":        False,
    "TEST_LAYER_LIMIT": 3,
    "TEST_ORANLAR":     [0.3],
}


def _detect_skip_patterns(yolo_model):
    # Ilk conv (model.0) ve Detect head'i otomatik tespit edip SKIP pattern uretir.
    # NOT: ".cv2." / ".cv3." pattern KULLANMA - C2f/C3k2 govde conv'larini da yakalar.
    seq = yolo_model.model  # nn.Sequential (top-level layer listesi)
    detect_idx = None
    for i, layer in enumerate(seq):
        if "Detect" in layer.__class__.__name__:
            detect_idx = i
    if detect_idx is None:
        detect_idx = len(seq) - 1  # Detect her zaman son katman
    skip = ("model.0.", f"model.{detect_idx}.", ".dfl.")
    print(f"[YOLO] Detect head -> model.{detect_idx}, SKIP={skip}")
    return skip


class C2f_v2(nn.Module):
    # YOLOv8 C2f, tek cv1 conv (out=2c) + chunk(2) kullanir. chunk torch_pruning
    # DepGraph'inde index/shape inference'i bozar (idx out-of-bounds /
    # "list index out of range"). C2f_v2 ayni hesabi chunk YERINE iki AYRI conv
    # (cv0, cv1) ile yapar -> graf temiz (yolo_fir_v5s'teki C3 gibi). Agirliklar
    # birebir tasinir, cikti aynidir. (VainF Torch-Pruning yolov8 recipe'i.)
    def __init__(self, c2f):
        super().__init__()
        from ultralytics.nn.modules import Conv
        c  = c2f.c
        c1 = c2f.cv1.conv.in_channels
        c2 = c2f.cv2.conv.out_channels
        n  = len(c2f.m)
        self.c   = c
        self.cv0 = Conv(c1, c, 1, 1)        # eski cv1'in ilk yarisi
        self.cv1 = Conv(c1, c, 1, 1)        # eski cv1'in ikinci yarisi
        self.cv2 = c2f.cv2                  # aynen
        self.m   = c2f.m                    # bottleneck'ler aynen
        self._transfer(c2f, c)
        self.to(next(c2f.parameters()).device)
        # Ultralytics routing attribute'lari (_predict_once m.f / m.i kullanir).
        self.f    = getattr(c2f, "f", -1)
        self.i    = getattr(c2f, "i", 0)
        self.type = getattr(c2f, "type", "C2f_v2")

    def _transfer(self, c2f, c):
        w = c2f.cv1.conv.weight.data
        self.cv0.conv.weight.data = w[:c].clone()
        self.cv1.conv.weight.data = w[c:].clone()
        if c2f.cv1.conv.bias is not None:
            self.cv0.conv.bias.data = c2f.cv1.conv.bias.data[:c].clone()
            self.cv1.conv.bias.data = c2f.cv1.conv.bias.data[c:].clone()
        for attr in ("weight", "bias", "running_mean", "running_var"):
            src = getattr(c2f.cv1.bn, attr).data
            getattr(self.cv0.bn, attr).data = src[:c].clone()
            getattr(self.cv1.bn, attr).data = src[c:].clone()
        self.cv0.bn.num_batches_tracked = c2f.cv1.bn.num_batches_tracked
        self.cv1.bn.num_batches_tracked = c2f.cv1.bn.num_batches_tracked

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _replace_c2f(model):
    # Modeldeki tum C2f'leri C2f_v2 ile degistir (recursive). v5s'te C2f yok -> 0.
    from ultralytics.nn.modules import C2f
    n = 0
    for name, child in model.named_children():
        if isinstance(child, C2f):
            setattr(model, name, C2f_v2(child))
            n += 1
        else:
            n += _replace_c2f(child)
    return n


def _load_yolo(path, device):
    # YOLO yukle + C2f'leri C2f_v2'ye cevir (pruning-uyumlu). Agirliklar korunur.
    yolo = YOLO(str(path))
    n = _replace_c2f(yolo.model)
    if n:
        print(f"[YOLO] {n} C2f -> C2f_v2 (chunk'siz, pruning-uyumlu).")
    yolo.model.to(device).eval()
    return yolo


def _collect_prunable(model, skip):
    # Prunable Conv2d katmanlarini topla (skip pattern degil, 1x1 degil, depthwise degil).
    names = []
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Conv2d):
            continue
        if any(p in name for p in skip):
            continue
        if layer.kernel_size == (1, 1):
            continue
        if layer.groups == layer.in_channels:
            continue  # depthwise -> atla
        names.append(name)
    return names


def prune_yolo(cfg: dict) -> dict:
    c = {**DEFAULTS, **cfg}
    for req in ("MODEL_PATH", "DATA_YAML", "OUT"):
        if not c[req]:
            raise ValueError(f"cfg['{req}'] zorunlu.")

    MODEL_PATH = str(c["MODEL_PATH"])
    DATA_YAML  = str(c["DATA_YAML"])
    IMG        = c["IMG"]
    DEVICE     = c["DEVICE"]
    OUT        = Path(c["OUT"]); OUT.mkdir(parents=True, exist_ok=True)

    oranlar  = c["ORANLAR"]
    layer_limit = None
    if c["TEST_MODE"]:
        oranlar = c["TEST_ORANLAR"]
        layer_limit = c["TEST_LAYER_LIMIT"]
        print("[TEST_MODE] aktif: oran=", oranlar, " katman_limit=", layer_limit)

    MIN_FLOOR = c["MIN_FLOOR"]
    MAX_CEIL  = c["MAX_CEIL"]

    # =====================================================================
    # 1) BASELINE
    # =====================================================================
    print("=" * 60); print("1) BASELINE"); print("=" * 60)

    yolo = _load_yolo(MODEL_PATH, DEVICE)

    baseline_map = None
    if not c["SKIP_BASELINE"]:
        s = yolo.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
        baseline_map = float(s.box.map)
        print("baseline mAP50-95 =", baseline_map)

    ornek_input = torch.randn(1, 3, IMG, IMG).to(DEVICE)
    baseline_macs, baseline_params = tp.utils.count_ops_and_params(yolo.model, ornek_input)
    baseline_real = count_params(yolo.model)
    print("params (real) =", baseline_real / 1e6, "M   MACs =", baseline_macs / 1e9, "G")

    skip = _detect_skip_patterns(yolo.model)
    prunable_isimler = _collect_prunable(yolo.model, skip)
    print("prunable katman sayisi:", len(prunable_isimler))

    if layer_limit is not None:
        sweep_isimler = prunable_isimler[:layer_limit]
    elif c["DO_SENSITIVITY"]:
        sweep_isimler = prunable_isimler
    else:
        sweep_isimler = []  # sweep yok -> tum katmanlara MIN_FLOOR

    # =====================================================================
    # 2) SENSITIVITY SWEEP
    # her katmana sirayla mask uygula -> mAP olc -> sifirla (model yeniden yuklenir)
    # =====================================================================
    sensitivity = {}
    if sweep_isimler:
        print("=" * 60); print("2) SENSITIVITY SWEEP"); print("=" * 60)
        for name in sweep_isimler:
            sensitivity[name] = {}
            for oran in oranlar:
                yolo_temp = _load_yolo(MODEL_PATH, DEVICE)
                modul_dict = dict(yolo_temp.model.named_modules())
                layer = modul_dict[name]
                nnp.ln_structured(layer, name="weight", amount=oran, n=2, dim=0)
                s = yolo_temp.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
                m = float(s.box.map)
                dusus = (baseline_map - m) if baseline_map is not None else 0.0
                sensitivity[name][oran] = dusus
                print(name, "oran=", oran, "mAP=", round(m, 4), "dusus=", round(dusus, 4))
                # KRITIK: del tek basina yetmez (YOLO<->validator<->dataloader ref
                # dongusu var); gc + empty_cache olmadan VRAM/RAM birikir.
                del yolo_temp, layer, modul_dict, s
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        with open(OUT / "sensitivity.json", "w") as f:
            json.dump(sensitivity, f, indent=2)
    else:
        print("[YOLO] sensitivity sweep atlandi -> tum katmanlara MIN_FLOOR uygulanacak.")

    # =====================================================================
    # 3) BUTCE: sensitivity'ye gore per-layer kesim orani
    # Az hassas katmani cok, cok hassas katmani az kes (MIN_FLOOR..MAX_CEIL).
    # =====================================================================
    print("=" * 60); print("3) BUTCE (sensitivity-ranked)"); print("=" * 60)
    skor = {n: sum(v.values()) / len(v) for n, v in sensitivity.items() if v}
    s_min = min(skor.values()) if skor else 0.0
    s_max = max(skor.values()) if skor else 0.0

    butce = {}
    for name in prunable_isimler:
        if name in skor and s_max > s_min:
            norm = (skor[name] - s_min) / (s_max - s_min)        # 0..1
            oran = MAX_CEIL - norm * (MAX_CEIL - MIN_FLOOR)
        else:
            oran = MIN_FLOOR
        oran = round(round(oran / 0.05) * 0.05, 2)               # 0.05'lik adimlar
        butce[name] = float(oran)
    with open(OUT / "butce.json", "w") as f:
        json.dump(butce, f, indent=2)
    print("ortalama hedef oran:", round(sum(butce.values()) / max(1, len(butce)), 3))

    # =====================================================================
    # 4) FIZIKSEL KESIM (DepGraph)
    # =====================================================================
    print("=" * 60); print("4) DepGraph ile fiziksel kesim"); print("=" * 60)
    yolo = _load_yolo(MODEL_PATH, DEVICE)
    ornek_input = torch.randn(1, 3, IMG, IMG).to(DEVICE)

    # KRITIK: ultralytics inference ckpt'ini parametreleri DONMUS yukler.
    # torch_pruning bagimlilik grafigini autograd grad_fn uzerinden kurar; grad
    # akmazsa graf baglanmaz (module2node=1) -> hicbir kanal kesilmez. Trace
    # oncesi gradi acmak grafigin tamamini olusturur.
    for p in yolo.model.parameters():
        p.requires_grad_(True)

    ignored_layers = [m for name, m in yolo.model.named_modules()
                      if isinstance(m, nn.Conv2d) and any(p in name for p in skip)]

    modul_dict = dict(yolo.model.named_modules())
    ratio_dict = {}
    for name in prunable_isimler:
        oran = max(float(butce.get(name, 0.0)), MIN_FLOOR)
        if oran > 0.0 and name in modul_dict:
            ratio_dict[modul_dict[name]] = oran
    print("efektif kesim hedefi:", len(ratio_dict), "katman")
    if not ratio_dict:
        raise RuntimeError(f"ratio_dict BOS -> kesim olmaz. MIN_FLOOR={MIN_FLOOR}")

    shape_before = {n: m.weight.shape for n, m in yolo.model.named_modules()
                    if isinstance(m, nn.Conv2d)}

    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        yolo.model,
        example_inputs=ornek_input,
        importance=importance,
        pruning_ratio=0.0,
        pruning_ratio_dict=ratio_dict,
        ignored_layers=ignored_layers,
        global_pruning=False,
        round_to=c["ROUND_TO"],          # kanallari cift tut -> C2f split((c,c)) bozulmaz
    )
    pruner.step()

    for m in yolo.model.modules():
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
    yolo.model.zero_grad(set_to_none=True)

    degisen = sum(1 for n, m in yolo.model.named_modules()
                  if isinstance(m, nn.Conv2d) and n in shape_before
                  and m.weight.shape != shape_before[n])
    print("shape degisen conv sayisi:", degisen)

    yeni_macs, yeni_params = tp.utils.count_ops_and_params(yolo.model, ornek_input)
    yeni_real = count_params(yolo.model)
    print("KESIM SONRASI params (real) =", yeni_real / 1e6, "M   MACs =", yeni_macs / 1e9, "G")
    print("azalma =", reduction_pct(baseline_real, yeni_real), "%")

    # =====================================================================
    # 5) FINE-TUNE (pruned modeli koruyarak)
    # =====================================================================
    if c["DO_FINETUNE"]:
        print("=" * 60); print("5) FINE-TUNE (pruned modeli koruyarak)"); print("=" * 60)
        from ultralytics.models.yolo.detect import DetectionTrainer

        # pruning scaffolding'i fine-tune'dan ONCE serbest birak (VRAM tirmanir).
        for _v in ("pruner", "importance"):
            if _v in locals():
                del locals()[_v]
        del shape_before
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        trainer = DetectionTrainer(overrides=dict(
            model=MODEL_PATH,        # sadece args/yaml icin; trainer.model'i altta eziyoruz
            data=DATA_YAML,
            epochs=c["FINETUNE_EPOCHS"],
            imgsz=IMG,
            lr0=c["LR0"],
            device=DEVICE,
            project=str(OUT),
            name="finetune",
            exist_ok=True,
            batch=c["BATCH"],
            workers=c["WORKERS"],
            cache=False,
        ))
        # DIKKAT: yolo.train() modeli yaml'dan YENIDEN kurar -> pruned yapi kaybolur.
        # trainer.model'i pruned nn.Module yapinca setup_model rebuild YAPMAZ.
        trainer.model = yolo.model
        trainer.train()
        yolo.model = trainer.model
        yolo.model.to(DEVICE).eval()
    else:
        print("[YOLO] fine-tune atlandi (DO_FINETUNE=False).")

    # =====================================================================
    # 6) FINAL EVAL + EXPORT
    # =====================================================================
    print("=" * 60); print("6) FINAL"); print("=" * 60)
    final_map = None
    if not c["SKIP_BASELINE"]:
        s = yolo.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
        final_map = float(s.box.map)
        print("mAP50-95:", baseline_map, "->", final_map,
              "delta=", (final_map - baseline_map) if baseline_map is not None else None)
    print("params (real):", baseline_real / 1e6, "M ->", yeni_real / 1e6, "M",
          " azalma=", reduction_pct(baseline_real, yeni_real), "%")

    # state_dict + tam model (pruned mimari rebuild edilemez, full obj lazim).
    pt_yol = OUT / "pruned_final.pt"
    torch.save({"model": yolo.model, "state_dict": yolo.model.state_dict()}, str(pt_yol))
    print("kaydedildi:", pt_yol)

    if c["DO_ONNX"]:
        # ONNX export DOGRUDAN pruned in-memory model'den (ultralytics yolo.export()
        # baseline ckpt'tan yeniden yukler -> pruned model ONNX'e gitmez).
        onnx_yol = OUT / "pruned_final.onnx"
        yolo.model.eval()
        torch.onnx.export(
            yolo.model, ornek_input, str(onnx_yol),
            opset_version=19,
            input_names=["images"], output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        )
        slim_onnx(onnx_yol)
        print("ONNX:", onnx_yol, " boyut:", onnx_yol.stat().st_size / 1e6, "MB")

    return {
        "baseline_map": baseline_map,
        "final_map": final_map,
        "params_before_M": baseline_real / 1e6,
        "params_after_M": yeni_real / 1e6,
        "reduction_pct": reduction_pct(baseline_real, yeni_real),
        "out": str(OUT),
    }

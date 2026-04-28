# YOLO pruning denemesi
# Hoca: structured pruning yap, sensitivity bak, fine-tune et, mAP karsilastir
# Kutuphane: torch_pruning (DepGraph) - kanali fiziksel kesiyor

import json
import torch
import torch.nn as nn
import torch.nn.utils.prune as nnp   # mask tabanli (sensitivity icin)
import torch_pruning as tp           # gercek kesim icin
from ultralytics import YOLO
from pathlib import Path


# ---- ayarlar ----
MODEL_PATH = "runs/sayzek_runs_yolo11_base_param/yolo11n_base_param12/weights/best.pt"
DATA_YAML = "/home/ugo/Documents/Python/uc_cihaz_obejct_detection/CNN-models-optimized-to-run-on-edge-devices/dataset/dataset_augmented_yolo_23.04.2026/dataset_augmented_yolo/data.yaml"
IMG = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("runs/pruning")
OUT.mkdir(parents=True, exist_ok=True)

# bunlara dokunmuyoruz: ilk conv (model.0), detect head (model.23, dfl, cv2, cv3)
SKIP = ("model.0.", "model.23", ".dfl", ".cv2.", ".cv3.")

# her katmana deneyecegimiz oranlar
ORANLAR = [0.1, 0.2, 0.3, 0.5]

# kac mAP dususu kabul ediyoruz
TOLERANS = 0.02


# ==========================================================
# 1) BASELINE
# ==========================================================
print("=" * 60)
print("1) BASELINE")
print("=" * 60)

yolo = YOLO(MODEL_PATH)
yolo.model.to(DEVICE).eval()

# baseline mAP
sonuc = yolo.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
baseline_map = float(sonuc.box.map)
print("baseline mAP50-95 =", baseline_map)

# param ve MAC sayisi
ornek_input = torch.randn(1, 3, IMG, IMG).to(DEVICE)
baseline_macs, baseline_params = tp.utils.count_ops_and_params(yolo.model, ornek_input)
print("params =", baseline_params / 1e6, "M")
print("MACs   =", baseline_macs / 1e9, "G")


# ==========================================================
# prunable katmanlari topla (Conv2d, skip pattern degil)
# ==========================================================
prunable_isimler = []
for name, layer in yolo.model.named_modules():
    if not isinstance(layer, nn.Conv2d):
        continue
    atla = False
    for p in SKIP:
        if p in name:
            atla = True
            break
    if atla:
        continue
    if layer.kernel_size == (1, 1):
        continue
    if layer.groups == layer.in_channels:
        # depthwise -> atla
        continue
    prunable_isimler.append(name)

print("prunable katman sayisi:", len(prunable_isimler))


# ==========================================================
# 2) SENSITIVITY SWEEP
# her katmana sirayla mask uygula -> mAP olc -> sifirla
# (mask geri alinabilir cunku her seferde modeli yeniden yukluyoruz)
# ==========================================================
print("=" * 60)
print("2) SENSITIVITY SWEEP")
print("=" * 60)

sensitivity = {}

for name in prunable_isimler:
    sensitivity[name] = {}
    for oran in ORANLAR:
        # modeli temizden yukle (onceki maskeler kalmasin)
        yolo_temp = YOLO(MODEL_PATH)
        yolo_temp.model.to(DEVICE).eval()

        # ismi modulle eslestir
        modul_dict = dict(yolo_temp.model.named_modules())
        layer = modul_dict[name]

        # L2 norm en dusuk %oran kanali sifirla (mask)
        nnp.ln_structured(layer, name="weight", amount=oran, n=2, dim=0)

        # mAP olc
        s = yolo_temp.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
        m = float(s.box.map)
        dusus = baseline_map - m
        sensitivity[name][oran] = dusus
        print(name, "oran=", oran, "mAP=", round(m, 4), "dusus=", round(dusus, 4))

        del yolo_temp
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# kaydet
with open(OUT / "sensitivity.json", "w") as f:
    json.dump(sensitivity, f, indent=2)


# ==========================================================
# 3) BUTCE: her katman icin dususu TOLERANS altinda kalan en yuksek oran
# ==========================================================
print("=" * 60)
print("3) BUTCE")
print("=" * 60)

butce = {}
for name in prunable_isimler:
    secilen = 0.0
    # kucukten buyuge dene, tolerans gecince dur
    for oran in sorted(sensitivity[name].keys()):
        if sensitivity[name][oran] <= TOLERANS:
            secilen = oran
        else:
            break
    butce[name] = secilen
    print(name, "->", secilen)

with open(OUT / "butce.json", "w") as f:
    json.dump(butce, f, indent=2)


# ==========================================================
# 4) FIZIKSEL KESIM (DepGraph)
# ==========================================================
print("=" * 60)
print("4) DepGraph ile fiziksel kesim")
print("=" * 60)

# temiz model yukle
yolo = YOLO(MODEL_PATH)
yolo.model.to(DEVICE).eval()

ornek_input = torch.randn(1, 3, IMG, IMG).to(DEVICE)
yolo.model.eval()

# skip listesindeki katmanlari ve detect head Conv'larini ignore et
ignored_layers = []
for name, m in yolo.model.named_modules():
    if isinstance(m, nn.Conv2d) and any(p in name for p in SKIP):
        ignored_layers.append(m)

# katman bazli oran sozlugu
modul_dict = dict(yolo.model.named_modules())
ratio_dict = {}
for name, oran in butce.items():
    if oran > 0 and name in modul_dict:
        ratio_dict[modul_dict[name]] = float(oran)

importance = tp.importance.MagnitudeImportance(p=2)

with torch.no_grad():
    pruner = tp.pruner.MagnitudePruner(
        yolo.model,
        example_inputs=ornek_input,
        importance=importance,
        pruning_ratio=0.0,
        pruning_ratio_dict=ratio_dict,
        ignored_layers=ignored_layers,
        global_pruning=False,
    )
    pruner.step()

print("kesim tamam, butce uygulandi:", len(ratio_dict), "katman")


# kesim sonrasi
yeni_macs, yeni_params = tp.utils.count_ops_and_params(yolo.model, ornek_input)
print("KESIM SONRASI params =", yeni_params / 1e6, "M")
print("KESIM SONRASI MACs   =", yeni_macs / 1e9, "G")


# ==========================================================
# 5) FINE-TUNE
# kesim sonrasi mAP duser, kucuk lr ile birkac epoch egit
# ==========================================================
#print("=" * 60)
#print("5) FINE-TUNE")
#print("=" * 60)

#yolo.train(
#    data=DATA_YAML,
#    epochs=10,
#    imgsz=IMG,
#    lr0=0.001,
#    device=DEVICE,
#    project=str(OUT),
#    name="finetune",
#    exist_ok=True,
#)


# ==========================================================
# 6) FINAL EVAL + EXPORT
# ==========================================================
print("=" * 60)
print("6) FINAL")
print("=" * 60)

s = yolo.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
final_map = float(s.box.map)

print("mAP50-95:", baseline_map, "->", final_map, "delta=", final_map - baseline_map)
print("params:", baseline_params / 1e6, "M ->", yeni_params / 1e6, "M")
print("MACs:  ", baseline_macs / 1e9, "G ->", yeni_macs / 1e9, "G")

# kaydet
pt_yol = OUT / "pruned_final.pt"
yolo.save(str(pt_yol))
print("kaydedildi:", pt_yol)

# ONNX export (Orin NX benchmark icin)
onnx_yol = yolo.export(format="onnx", imgsz=IMG, simplify=True)
print("ONNX:", onnx_yol)

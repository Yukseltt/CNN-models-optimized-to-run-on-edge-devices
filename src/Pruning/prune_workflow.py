# YOLO pruning denemesi
# Hoca: structured pruning yap, sensitivity bak, fine-tune et, mAP karsilastir
# Kutuphane: torch_pruning (DepGraph) - kanali fiziksel kesiyor

import os
import gc
import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
# bu host'ta DataLoader worker'lari + paylasimli bellek FD'leri tuketip EMFILE veriyor;
# fine-tune egitiminden once file_system stratejisi sart (yoksa egitim takilir/coker).
mp.set_sharing_strategy("file_system")
import torch.nn.utils.prune as nnp   # mask tabanli (sensitivity icin)
import torch_pruning as tp           # gercek kesim icin
from ultralytics import YOLO
from pathlib import Path


# ---- ayarlar ----
MODEL_PATH = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs/yolo_fir_v5s_28.05.2026/weights/best.pt"
DATA_YAML = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/2x_augmented_yolo_dataset/dataset_augmented_yolo/data.yaml"
IMG = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs_pruing")
OUT.mkdir(parents=True, exist_ok=True)

# bunlara dokunmuyoruz: ilk conv (model.0) ve detect head (model.24)
# NOT: bu model yolov5su -> Detect head index'i model.24 (yolo11'de model.23 idi)
# NOT: ".cv2." / ".cv3." pattern KULLANMA - C2f/C3k2 govde conv'larini da yakalar
SKIP = ("model.0.", "model.24.", ".dfl.")

# her katmana deneyecegimiz oranlar
ORANLAR = [0.1, 0.2, 0.3, 0.5]

# Her prunable katman icin GARANTI minimum kesim orani.
# DIKKAT: 0.0 + "drop <= tolerans" mantigi = HICBIR sey kesilmez (model kuculmez),
# cunku structured pruning fine-tune'dan ONCE mAP'i her zaman cok dusurur.
# Fine-tune ile telafi edecegimiz icin tabani >0 tut (0.10-0.20 mantikli).
MIN_FLOOR = 0.10
# Az hassas katmana izin verilen en yuksek oran (sensitivity'ye gore MIN_FLOOR..MAX_CEIL).
MAX_CEIL = 0.40

# ---- TEST MODE ----
# Kisa duman testi: az katman + tek oran + val subset. Kapatmak icin asagidaki satiri yorum yap.
TEST_MODE = False   # <-- kapatmak icin bu satiri yorumla (veya False yap)

if "TEST_MODE" in globals() and TEST_MODE:
    ORANLAR = [0.3]              # tek oran
    TEST_LAYER_LIMIT = 3         # sensitivity sweep'te ilk N prunable katman
    print("[TEST_MODE] aktif: oran=", ORANLAR, " katman_limit=", TEST_LAYER_LIMIT)
else:
    TEST_LAYER_LIMIT = None


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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
baseline_real = count_params(yolo.model)
print("params (tp)   =", baseline_params / 1e6, "M")
print("params (real) =", baseline_real / 1e6, "M")
print("MACs          =", baseline_macs / 1e9, "G")


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

# TEST_MODE: sensitivity sweep'i ilk N katmana dustur (tum kesim listesi degil)
if TEST_LAYER_LIMIT is not None:
    sweep_isimler = prunable_isimler[:TEST_LAYER_LIMIT]
    print("[TEST_MODE] sweep katman sayisi:", len(sweep_isimler))
else:
    sweep_isimler = prunable_isimler


# ==========================================================
# 2) SENSITIVITY SWEEP
# her katmana sirayla mask uygula -> mAP olc -> sifirla
# (mask geri alinabilir cunku her seferde modeli yeniden yukluyoruz)
# ==========================================================
print("=" * 60)
print("2) SENSITIVITY SWEEP")
print("=" * 60)

sensitivity = {}

for name in sweep_isimler:
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

        # KRITIK: del tek basina yetmez. YOLO modeli <-> validator <-> dataloader
        # arasinda ref dongusu var; saf refcount bunu hemen bosaltmaz -> her
        # iterasyonda VRAM/RAM birikir ("vram surekli artiyor"). gc.collect()
        # donguleri kirar, empty_cache() suruculuye geri verir.
        del yolo_temp, layer, modul_dict, s
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# kaydet
with open(OUT / "sensitivity.json", "w") as f:
    json.dump(sensitivity, f, indent=2)


# ==========================================================
# 3) BUTCE: sensitivity'ye gore per-layer kesim orani
# ==========================================================
# NEDEN ESKI KOD MODELI KUCULTMUYORDU:
#   Eski mantik "mAP dususu <= TOLERANS (0.02) olan en buyuk orani sec" idi.
#   Ama structured pruning fine-tune'dan ONCE her zaman mAP'i ciddi dusurur:
#   tek katmani %30 kesince mAP 0.30-0.57 dusuyor (sensitivity.json). Hicbir
#   katman 0.02 esiginin altina inmedigi icin butce HER katmana 0.0 veriyordu
#   -> ratio_dict bos -> torch_pruning hicbir kanali kesmiyor -> AYNI boyut.
#
#   Dogru yaklasim: sensitivity ile katmanlari SIRALA, garanti taban (MIN_FLOOR)
#   ile kes, sonra fine-tune ile mAP'i geri kazan. Az hassas katmani cok, cok
#   hassas katmani az kes (MIN_FLOOR..MAX_CEIL araliginda).
print("=" * 60)
print("3) BUTCE (sensitivity-ranked)")
print("=" * 60)

# sensitivity skoru: swept katmanlarin ortalama mAP dususu (yuksek = daha hassas)
skor = {n: sum(v.values()) / len(v) for n, v in sensitivity.items() if v}
s_min = min(skor.values()) if skor else 0.0
s_max = max(skor.values()) if skor else 0.0

butce = {}
for name in prunable_isimler:
    if name in skor and s_max > s_min:
        # az hassas (dusuk skor) -> MAX_CEIL'e yakin; cok hassas -> MIN_FLOOR'a yakin
        norm = (skor[name] - s_min) / (s_max - s_min)          # 0..1
        oran = MAX_CEIL - norm * (MAX_CEIL - MIN_FLOOR)
    else:
        # sweep edilmemis (TEST_MODE) katman -> guvenli taban
        oran = MIN_FLOOR
    oran = round(round(oran / 0.05) * 0.05, 2)                 # 0.05'lik adimlara yuvarla
    butce[name] = float(oran)
    skor_str = f"{skor[name]:.3f}" if name in skor else "n/a"
    print(f"{name} -> {oran:.2f}  (sensitivity skoru={skor_str})")

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

# KRITIK: ultralytics inference ckpt'ini parametreleri DONMUS (requires_grad=False)
# yukler. torch_pruning bagimlilik grafigini autograd grad_fn uzerinden kurar; grad
# akmazsa graf baglanmaz (module2node=1) -> get_all_groups()=0 -> hicbir kanal kesilmez
# ("shape degisen conv: 0"). Trace oncesi gradi acmak grafigin tamamini olusturur.
for p in yolo.model.parameters():
    p.requires_grad_(True)

# skip listesindeki katmanlari ve detect head Conv'larini ignore et
ignored_layers = []
for name, m in yolo.model.named_modules():
    if isinstance(m, nn.Conv2d) and any(p in name for p in SKIP):
        ignored_layers.append(m)

# katman bazli oran sozlugu. Sensitivity=0.0 ise kesim yok.
modul_dict = dict(yolo.model.named_modules())
ratio_dict = {}
non_zero = 0
for name in prunable_isimler:
    oran = max(float(butce.get(name, 0.0)), MIN_FLOOR)
    if oran <= 0.0:
        continue  # sensitivity hassas dedi, dokunma
    if name in modul_dict:
        ratio_dict[modul_dict[name]] = oran
        non_zero += 1

print("efektif kesim hedefi:", non_zero, "katman")
if non_zero == 0:
    raise RuntimeError(
        "ratio_dict BOS -> hicbir kanal kesilmeyecek, model kuculmez. "
        f"MIN_FLOOR > 0 oldugundan emin ol (su an {MIN_FLOOR})."
    )

# kesim oncesi conv shape snapshot (dogrulama)
shape_before = {n: m.weight.shape for n, m in yolo.model.named_modules()
                if isinstance(m, nn.Conv2d)}

importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    yolo.model,
    example_inputs=ornek_input,
    importance=importance,
    pruning_ratio=0.0,                  # global default sifir
    pruning_ratio_dict=ratio_dict,      # per-layer override
    ignored_layers=ignored_layers,
    global_pruning=False,
    round_to=1,                         # kanal yukari yuvarlamayi engelle
)
pruner.step()

# Pruner state temizle - count_ops_and_params dogru sayim icin
for m in yolo.model.modules():
    if hasattr(m, "_forward_pre_hooks"):
        m._forward_pre_hooks.clear()
yolo.model.zero_grad(set_to_none=True)

print("kesim tamam, butce uygulandi:", len(ratio_dict), "katman, MIN_FLOOR=", MIN_FLOOR)

# shape diff - gercekten kesilmis mi dogrula
degisen = 0
ornekler = []
for n, m in yolo.model.named_modules():
    if isinstance(m, nn.Conv2d) and n in shape_before:
        if m.weight.shape != shape_before[n]:
            degisen += 1
            if len(ornekler) < 5:
                ornekler.append((n, tuple(shape_before[n]), tuple(m.weight.shape)))
print("shape degisen conv sayisi:", degisen)
for n, a, b in ornekler:
    print("  ", n, a, "->", b)

# kesim sonrasi
yeni_macs, yeni_params = tp.utils.count_ops_and_params(yolo.model, ornek_input)
yeni_real = count_params(yolo.model)
print("KESIM SONRASI params (tp)   =", yeni_params / 1e6, "M")
print("KESIM SONRASI params (real) =", yeni_real / 1e6, "M")
print("KESIM SONRASI MACs          =", yeni_macs / 1e9, "G")


# ==========================================================
# 5) FINE-TUNE
# kesim sonrasi mAP duser, kucuk lr ile birkac epoch egit
# ==========================================================
# DIKKAT: yolo.train() modeli self.model.yaml'dan YENIDEN kurar (get_model).
# YOLO yaml'i depth/width carpanlari + modul listesi tutar, kesilmis kanal
# sayilarini DEGIL -> pruned yapi KAYBOLUR, model 9.12M'e geri doner ve kesim
# bosa gider. Cozum: trainer'i manuel kur, trainer.model'i pruned nn.Module yap.
# DetectionTrainer.setup_model "isinstance(model, nn.Module) -> return" oldugu icin
# yeniden kurmaz; pruned modeli egitir.
print("=" * 60)
print("5) FINE-TUNE (pruned modeli koruyarak)")
print("=" * 60)

from ultralytics.models.yolo.detect import DetectionTrainer

# KRITIK: pruning scaffolding'i fine-tune'dan ONCE serbest birak.
# pruner tum DependencyGraph + modeli pinler; importance ve trace artiklari
# da VRAM tutar. Silmezsek bu bellek egitim boyunca geri verilmez ve egitimin
# kendi train+EMA+val tahsisinin USTUNE biner -> VRAM tirmanir.
for _v in ("pruner", "importance", "shape_before"):
    if _v in globals():
        del globals()[_v]
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

trainer = DetectionTrainer(overrides=dict(
    model=MODEL_PATH,        # sadece args/yaml icin; trainer.model'i altta eziyoruz
    data=DATA_YAML,
    epochs=10,
    imgsz=IMG,
    lr0=0.001,
    device=DEVICE,
    project=str(OUT),
    name="finetune",
    exist_ok=True,
    batch=8,        # 16 -> 8
    workers=4,      # 8 -> 4 (val worker'larıyla toplam RAM zirvesini düşürür)
    cache=False,    # zaten kapalı, açık bırakma
))
trainer.model = yolo.model   # pruned nn.Module -> setup_model rebuild YAPMAZ
trainer.train()

# egitilen (pruned) modeli geri al; sonraki val/save/export bunu kullansin
yolo.model = trainer.model
yolo.model.to(DEVICE).eval()


# ==========================================================
# 6) FINAL EVAL + EXPORT
# ==========================================================
print("=" * 60)
print("6) FINAL")
print("=" * 60)

s = yolo.val(data=DATA_YAML, imgsz=IMG, device=DEVICE, verbose=False)
final_map = float(s.box.map)

print("mAP50-95:", baseline_map, "->", final_map, "delta=", final_map - baseline_map)
print("params (tp)  :", baseline_params / 1e6, "M ->", yeni_params / 1e6, "M")
print("params (real):", baseline_real / 1e6, "M ->", yeni_real / 1e6, "M",
      "  azalma=", round((1 - yeni_real / baseline_real) * 100, 2), "%")
print("MACs         :", baseline_macs / 1e9, "G ->", yeni_macs / 1e9, "G")

# kaydet (state_dict + tam model - pruned mimari rebuild edilemez, full obj lazim)
pt_yol = OUT / "pruned_final.pt"
torch.save({"model": yolo.model, "state_dict": yolo.model.state_dict()}, str(pt_yol))
print("kaydedildi:", pt_yol)

# ONNX export DOGRUDAN pruned in-memory model'den (ultralytics yolo.export() baseline
# ckpt'tan yeniden yukler -> pruned model ONNX'e gitmez. torch.onnx.export ile bypass.)
onnx_yol = OUT / "pruned_final.onnx"
yolo.model.eval()
torch.onnx.export(
    yolo.model,
    ornek_input,
    str(onnx_yol),
    opset_version=19,
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
)

# onnxsim ile slim
try:
    import onnx
    import onnxruntime as ort
    from onnxsim import simplify

    # onnxsim icindeki onnxruntime, intra_op_num_threads belirtilmediginde worker
    # thread'leri sirali CPU indekslerine (0..N-1) pinlemeye calisir. Bu host'ta
    # izinli cekirdekler non-contiguous ({4,6,21,...}) oldugundan
    # "pthread_setaffinity_np failed ... error code: 22" hatasi basar.
    # Cozum (ORT mesajinin kendi onerisi): thread sayisini ACIKCA ver -> ORT affinity
    # pinlemeyi tamamen atlar. onnxsim kendi SessionOptions'ini (intra_op=0/auto)
    # uretir; o yuzden InferenceSession'i kisa sureligine sarmalayip degeri set ediyoruz.
    _n_threads = max(1, len(os.sched_getaffinity(0)))
    _orig_session = ort.InferenceSession

    def _session_with_threads(*args, **kwargs):
        so = kwargs.get("sess_options")
        if so is None and len(args) >= 2 and isinstance(args[1], ort.SessionOptions):
            so = args[1]
        if so is not None and so.intra_op_num_threads == 0:
            so.intra_op_num_threads = _n_threads
        return _orig_session(*args, **kwargs)

    ort.InferenceSession = _session_with_threads
    try:
        m = onnx.load(str(onnx_yol))
        m_simp, ok = simplify(m)
    finally:
        ort.InferenceSession = _orig_session  # patch'i geri al

    if ok:
        onnx.save(m_simp, str(onnx_yol))
        print("onnxsim: ok")
except Exception as e:
    print("onnxsim atlandi:", e)

print("ONNX:", onnx_yol)
print("ONNX boyutu:", onnx_yol.stat().st_size / 1e6, "MB")

# edge_perf — Uç cihaz performans benchmark'ı (tek `main.py`)

5 modelin **base (düz) / pruned / quantize (fp16 + int8)** hâllerini, çalıştığı platformu
otomatik algılayarak ölçer; latency/FPS, mAP@0.5 / mAP@0.5:0.95, güç/enerji (cihazda),
disk, param ve türetilmiş verimlilik metriklerini çıkarır, **kaydeder ve sıralar**.

Tek komut:

```bash
python edge_perf/main.py
```

Platformu kendi algılar:

| Platform | Backend | normal / pruned | quant_fp16 | quant_int8 |
|---|---|---|---|---|
| **server** (H200/CUDA) | torch | fp32 | fp16 (GPU) | — (torch'ta yok, SKIP) |
| **jetson** (Orin Nano) | TensorRT | fp16 engine | fp16 engine | **int8** engine (kalibre) |
| **rpi5_hailo** (Pi5 + Hailo-8L) | HailoRT / onnxruntime-CPU | fp32 onnx | fp16 onnx | **int8** (.hef varsa Hailo NPU, yoksa int8 onnx CPU) |

Çıktı: `edge_perf/results/edge_perf_<platform>_<ts>.{json,csv}` + `_ranking.txt` (leaderboard).

---

## 1) Kapsanan 5 model

| key | tip | imgsz | base | pruned |
|---|---|---|---|---|
| `yolov8n` | yolo | 640 | ✔ | ✗ *(finetune'lu pruned ağırlık yok — bkz. Notlar)* |
| `yolo_fir_v5s` | yolo | 640 | ✔ | ✔ (finetune'lu) |
| `faster_rcnn_resnet50` | frcnn | 800 | ✔ | ✔ |
| `faster_rcnn_mobilenet` | frcnn | 800 | ✔ | ✔ |
| `rtdetr_v2_r18vd` | rtdetr | 480 | ✔ | ⚠ (finetune'suz, mAP≈0 — bkz. Notlar) |

Tüm model yolları ve platform matrisi **`config.py`** içindedir; başka yerde model yolu yoktur.

---

## 2) `models/` klasörü (tek-paket, taşınabilir)

Tüm ağırlık/artifact'lar `edge_perf/models/<key>/` altında, model başına:

```
models/<key>/
  base.pt        pruned.pt            # kaynak (PyTorch)
  base.onnx      pruned.onnx          # fp32 ONNX
  base_fp16.onnx pruned_fp16.onnx     # fp16 ONNX  (quant_fp16)
  base_int8.onnx pruned_int8.onnx     # int8 ONNX  (quant_int8, onnxruntime-CPU)
  base_fp16.engine ...                # Jetson TensorRT (cihazda üretilir)
  base.hef       pruned.hef           # Hailo-8L int8 (x86 Hailo DFC ile üretilir)
```

`.gitignore` ağırlıkları repodan dışlar; kodu paylaşır, ağırlıkları aşağıdaki adımlarla üretirsin.

### Hazırlık (sunucuda, bir kez)
```bash
python edge_perf/prepare_models.py            # base/pruned .pt -> models/<key>/
python edge_perf/convert/export_onnx.py       # -> fp32 .onnx  (frcnn ONNX kırılgan, atlanabilir)
python edge_perf/convert/quantize_onnx.py     # -> _fp16.onnx + _int8.onnx
```

---

## 3) Raspberry Pi 5 + Hailo-8L'de çalıştırma

> Amaç: **klasörü Pi'ye atınca çalışsın.** Hailo `.hef` yoksa bile onnxruntime-CPU
> ile base/pruned/quant_fp16/quant_int8 ölçülür; `.hef` varsa int8 otomatik NPU'da koşar.

```bash
# 1) klasörü Pi'ye kopyala (models/ dahil)
scp -r edge_perf pi@<ip>:~

# 2) Pi'de bağımlılıklar
python3 -m venv venv && source venv/bin/activate
pip install -r edge_perf/requirements_rpi5.txt

# 3) çalıştır (platformu Raspberry olarak algılar)
python edge_perf/main.py
#   sadece hız (mAP atla):           python edge_perf/main.py --no-map
#   hızlı duman testi:               python edge_perf/main.py --smoke
```

### Hailo `.hef` (int8) — en iyi NPU performansı için (x86 dev makinede)
Hailo NPU yalnız **int8** çalışır; `.hef` ancak Hailo Dataflow Compiler (x86) ile, kalibrasyon
görüntüleriyle derlenir — Pi'de veya bu sunucuda üretilemez.
```bash
# x86 + Hailo DFC kurulu makinede:
mkdir -p edge_perf/models/yolov8n/calib_imgs   # ~100+ temsili termal jpg koy
bash edge_perf/convert/build_hailo.sh yolov8n_base     # (export_onnx.py önce çalışmış olmalı)
bash edge_perf/convert/build_hailo.sh yolo_fir_v5s_pruned
# çıkan .hef -> models/<key>/ ; Pi'ye kopyala. main.py int8 satırında otomatik kullanır.
```
frcnn (two-stage) ve rtdetr (transformer) Hailo-8L'de pratikte derlenmez → Pi'de
**onnxruntime-CPU** (frcnn için ONNX yoksa **torch-CPU**) ile ölçülür.

---

## 4) Jetson Orin Nano'da çalıştırma

```bash
sudo jetson_clocks && sudo nvpmodel -m 0          # saat sabitle (tutarlı ölçüm)
# YOLO engine (fp16 + int8):
python edge_perf/convert/build_tensorrt.py        # -> models/<key>/<lbl>_<prec>.engine
# frcnn/rtdetr: export_onnx.py'nin ONNX'i; onnxruntime-TRT EP ilk run'da engine'i JIT derler
python edge_perf/main.py                           # platformu Jetson algılar
```

---

## 5) "Hangi quantize?" — yorum

- **Jetson:** `quant_fp16` vs `quant_int8` satırlarını karşılaştır. INT8 ~2x hız + ~4x küçük
  disk verir; mAP düşüşü kabul edilebilirse (genelde birkaç puan) **int8** tercih et, kritik
  doğrulukta **fp16**.
- **Raspberry Pi + Hailo-8L:** NPU **zorunlu int8** → dağıtım için `.hef` (int8) tek mantıklı yol;
  fp16/CPU yalnız Hailo yokken yedek. Yani Pi+Hailo'da cevap pratikte **int8**.
- INT8 mAP düşüşü, pruning mAP düşüşünün **üstüne biner**; her zaman fp32 (server) base'i ile
  karşılaştır. Pruning'i finetune'la kurtarmadan int8'e gitme.

---

## 6) Faydalı bayraklar
```
--platform {auto,server,jetson,rpi5_hailo}   # algılamayı ez
--models yolov8n,rtdetr_v2_r18vd             # alt küme
--variants normal,quant_int8                 # alt küme
--map-images 200                             # mAP'i hızlı alt-kümede
--no-map                                     # sadece hız
--smoke                                      # 5 warmup / 20 ölçüm / 50 mAP görüntü
--conf 0.001                                 # mAP eşiği (DETR ailesi için düşük şart)
```
Ortam değişkeni: `EDGE_PLATFORM=server|jetson|rpi5_hailo` ile platformu zorla.

---

## Notlar / bilinen durumlar
- **Pruned ağırlıklar finetune'suz üretilmiş** (`prune_*` wrapper'larında `DO_FINETUNE=False`):
  - `yolo_fir_v5s` finetune'lu pruned = `runs_pruing/finetune/weights/best.pt` (config bunu kullanır, sağlam).
  - `rtdetr_v2_r18vd` pruned_final.pt **finetune'suz → mAP≈0**; anlamlı sonuç için finetune şart.
  - `yolov8n` pruned çıktısı yok (`runs_pruing/yolov8n` boş) → pruned/quant satırları SKIP.
  - frcnn pruned (backbone-only, hafif) finetune'suz da makul kalıyor.
- **mAP eşiği:** RT-DETR/DETR düşük-kalibre skor üretir; `--conf` düşük (0.001) olmalı, yoksa mAP=0.
- **rtdetr imgsz=480** (eğitim boyutu); 640 verilirse güven ~0'a düşer.
- **onnxruntime:** Pi'de **CPU ARM** wheel kullan (`onnxruntime`), `onnxruntime-gpu` değil.

# Pruning Pipeline — Özet

Bu doküman, `src/Pruning/` altındaki structured pruning çalışmasını özetler.
Amaç: benchmark'taki **en başarılı ve birbirinden farklı 5 modeli** budayıp (structured
channel pruning) uç cihaz (Jetson / RPi5) dağıtımına hazırlamak.

## Budanan 5 model

| Model | Aile / Mimari | Backbone | baseline | Strateji |
|---|---|---|---|---|
| faster_rcnn_resnet50_28.04.2026 | Two-stage (Faster R-CNN) | ResNet50 | mAP@0.5≈0.77 | backbone-only + sweep |
| rtdetr_v2_r18vd_from_scratch_26.05.2026 | Transformer (DETR) | ResNet18 | — | backbone-only + sweep |
| faster_rcnn_mobilenet_scratch_12.05.2026 | Two-stage | MobileNetV3 | — | backbone-only + sweep |
| yolo_fir_v5s_28.05.2026 | One-stage (YOLO/C3) | CSP v5s | mAP50-95≈0.57 | tam sweep + DepGraph |
| yolov8n_default_augmented_29.04.2026 | One-stage (YOLO/C2f) | v8n | — | tam sweep + DepGraph |

Kütüphane: **torch_pruning** (DepGraph — kanalı fiziksel keser). Fine-tune dataseti:
modellerin eğitildiği eski `2x_augmented` (coco/yolo). Çıktı: `runs_pruing/<model>/`.

## Dosya yapısı

```
src/Pruning/
├── prune_common.py            # ortak: mp-sharing, param sayacı, build_budget, onnxsim slim
├── prune_lib_yolo.py          # prune_yolo(cfg)   — YOLO tam pipeline
├── prune_lib_frcnn.py         # prune_frcnn(cfg)  — Faster R-CNN backbone-only
├── prune_lib_rtdetr.py        # prune_rtdetr(cfg) — RT-DETR backbone-only
├── prune_yolo_fir_v5s.py      # entry script'ler (her biri CFG + lib çağrısı)
├── prune_yolov8n.py
├── prune_faster_rcnn_resnet50.py
├── prune_faster_rcnn_mobilenet.py
├── prune_rtdetr_v2_r18vd.py
└── prune_workflow.py          # eski tek-YOLO script (referans, dokunulmadı)
```

Her entry script `smoke` argümanı destekler: `python prune_yolov8n.py smoke`
(sadece load→prune→kaydet; sweep/eval/fine-tune yok — hızlı sağlık kontrolü).

## Pipeline adımları

1. **BASELINE** — modeli yükle, val mAP ölç.
2. **SENSITIVITY SWEEP** — her prunable katmana **tek tek** mask uygula
   (`%10/20/30/50`), mAP düşüşüne bak. (frcnn/rtdetr için val **subset**'inde,
   `SWEEP_VAL_IMAGES=200`; full val çok yavaş.)
3. **PER-LAYER BÜTÇE** (`build_budget`) — hassas katman `MIN_FLOOR=0.10` (az kes),
   dayanıklı katman `MAX_CEIL=0.40` (çok kes). `sensitivity.json` + `butce.json` kaydedilir.
4. **FİZİKSEL KESİM** — DepGraph ile per-layer `pruning_ratio_dict` uygulanır.
5. **VERIFY** — kesilmiş modelle tam forward (fail-fast).
6. **FINE-TUNE** — küçük lr ile yeniden eğit *(şu an KAPALI — aşağıya bak)*.
7. **FINAL** — kesim-sonrası mAP + kaydet (`pruned_final.pt`, YOLO ayrıca `.onnx`).

### Mimari farkları
- **YOLO**: tüm gövde (3×3, 1×1 değil, depthwise değil) budanır; Detect head + ilk
  conv + dfl korunur (auto-detect: v5s→`model.24`, v8n→`model.22`).
- **Faster R-CNN**: `GeneralizedRCNN.forward` trace edilemez → sadece `model.backbone`
  budanır, **FPN çıkışı 256 sabit** (RPN/ROI head'ler bozulmaz).
- **RT-DETR**: sadece ResNet18 backbone budanır; `encoder_input_proj` ignored →
  **d_model=256 sabit**, transformer encoder/decoder dokunulmaz.

## Çözülen kritik sorunlar (torch_pruning gotcha'ları)

1. **FrozenBatchNorm2d** (torchvision + RT-DETR backbone): weight/bias/running_*
   hepsi *buffer*, torch_pruning tanımıyor → conv budanınca BN küçülmüyor
   (`size a(44) vs b(64)`). Çözüm: özel `FrozenBNPruner` + `customized_pruners`.
2. **YOLOv8 C2f** tek conv + `chunk(2)` → DepGraph index/shape inference patlıyor.
   Çözüm: C2f'i iki ayrı conv'lu **C2f_v2**'ye çevir (ağırlıklar split edilerek
   taşınır; `f/i/type` routing attr'ları kopyalanır). v5s (C3) bu sorundan etkilenmez.
3. **Sweep subset eval deflate**: frcnn'de COCOeval tüm val GT'si üzerinden ölçüyor →
   subset tahminleri ~100× deflate, sensitivity sinyali ölüyor. Çözüm:
   `coco_eval.params.imgIds = subset_ids`.
4. **bf16_full_eval=True**: HF `trainer.evaluate()` sonrası modeli kalıcı bf16 yapıp
   sonraki prune/verify'ı patlatıyor. Çözüm: `bf16_full_eval=False` + prune öncesi `model.float()`.
5. **mobilenet**: prunable 3×3 kök sadece **1** (gerisi 1×1/depthwise) → sweep
   neredeyse no-op. Daha çok kesim için `DO_SENSITIVITY=False` (uniform, 1×1 dahil).

## Sonuçlar (fine-tune KAPALI — ham budama hasarı)

| Model | Param (önce→sonra) | Azalma | baseline mAP → kesim-sonrası |
|---|---|---|---|
| faster_rcnn_resnet50 (sweep) | 41.31M → 34.76M | **%15.9** | mAP@0.5 0.768 → **0.251** |
| yolo_fir_v5s (sweep) | 9.11M → 7.24M | **%20.5** | mAP50-95 0.569 → **0.0005** |
| faster_rcnn_resnet50 (uniform %30, smoke) | 41.31M → 29.02M | %29.7 | — |
| faster_rcnn_mobilenet (uniform, smoke) | 18.94M → 17.37M | %8.3 | — |
| rtdetr_v2_r18vd (uniform, smoke) | 20.08M → 14.28M | %28.9 | — |

> **Önemli:** Structured pruning fine-tune **öncesi** mAP'ı her zaman ciddi düşürür
> (yolo_fir 0.569→0.0005, frcnn 0.77→0.25). Bu beklenen davranıştır — sweep, hassas
> katmanı (örn. frcnn stem `body.conv1` → skor 0.264) %10'da koruyup dayanıklı derin
> katmanları %40 keserek hasarı en aza indirir. **Kullanılabilir doğruluk için
> fine-tune şart.**

### Sweep'in çalıştığının kanıtı (frcnn_resnet50, 17 katman)
- `body.conv1` (stem) → sensitivity skoru **0.264** (en hassas) → oran **0.10**
- 15 derin `conv2` → skor 0.002–0.024 (dayanıklı) → oran **0.40**
- `body.layer2.0.conv2` → orta → 0.35

## Çalıştırma

```bash
cd /home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection
source venv/bin/activate
cd src/Pruning

# tek model
python prune_faster_rcnn_resnet50.py 2>&1 | tee runlog_frcnn_r50.txt

# hepsi sırayla (H200 MIG küçük → sıralı, paralel değil)
for s in prune_faster_rcnn_resnet50 prune_faster_rcnn_mobilenet \
         prune_rtdetr_v2_r18vd prune_yolo_fir_v5s prune_yolov8n; do
    python "$s.py" 2>&1 | tee "runlog_$s.txt"
done
```

### Ayarlar (entry script CFG'sinde)
- `DO_FINETUNE`: şu an **False** (ham hasar görmek için). Doğruluğu geri kazanmak
  için `True` yap.
- `DO_SENSITIVITY`: True (sweep). `False` → uniform `PRUNE_RATIO`.
- `MIN_FLOOR` / `MAX_CEIL`: kesim oranı tabanı/tavanı (varsayılan 0.10 / 0.40).
- `SWEEP_VAL_IMAGES`: sweep subset boyutu (frcnn/rtdetr, varsayılan 200).
- YOLO sweep full val'de → saatler sürer; hızlandırmak için `DO_SENSITIVITY=False`.

## Süre notları
- frcnn_resnet50 (sweep, 200-img subset): ~30–45 dk
- frcnn_mobilenet: hızlı (1 prunable kök)
- rtdetr: ~20–30 dk
- yolo_fir_v5s / yolov8n: sweep full val'de → **saatler**

## Sonraki adım
Budanan modeller (`runs_pruing/<model>/pruned_final.pt` + `.onnx`) → uç cihaz
benchmark'ı (`src/edge_benchmark/`): Jetson Orin Nano (TensorRT) + RPi5 + Hailo-8L.
TensorRT (FP16/INT8) ve Hailo (INT8) kuantizasyonu, pruning hasarının **üstüne** biner.

# Edge Benchmark — Jetson Orin Nano & Raspberry Pi 5 + AI HAT (Hailo-8L)

Pruned (ve istersen quantize) modelleri iki uç cihazda ölçer.

## Ölçülen metrikler
- **Hız**: latency mean/median/p90/p95/p99/std/min/max (ms), FPS (steady-state, 20 warmup + 100 ölçüm, batch=1)
- **Güç & enerji**: ortalama/tepe güç (W), **enerji/frame (mJ)** = W×ms, **FPS/Watt**
- **Kaynak**: peak RAM (MB), GPU/NPU util, sıcaklık (°C), throttling (Pi5), disk (MB)
- **Doğruluk**: mAP@0.5, mAP@0.5:0.95, per-class AP@0.5 — **kuantizasyon sonrası** (FP16/INT8)
- **Türetilmiş**: FPS/W, mAP/W, mAP/milyon-param

Çıktı: `bechmarks/jetson_orin_nano_<ts>.{json,csv}` ve `bechmarks/rpi5_hailo8l_<ts>.{json,csv}`.

## Dosyalar
- `bench_core.py` — cihaz-bağımsız çekirdek (stats, sampler, mAP, rapor)
- `benchmark_jetson_orin_nano.py` — TensorRT (YOLO: ultralytics .engine; frcnn/rtdetr: onnxruntime-TRT EP)
- `benchmark_rpi5_hailo8l.py` — HailoRT .hef (+ Hailo'ya derlenemeyenler için Pi5 CPU onnxruntime baseline)

Her iki script'te en üstteki **`REGISTRY`** listesindeki `artifact` yollarını cihazındaki dosyalara göre düzenle.

---

## 1) Model dönüştürme (ÖNKOŞUL)

### YOLO → TensorRT (Jetson)
```bash
# cihazda (ultralytics kurulu):
yolo export model=runs_pruing/yolov8n/finetune/weights/best.pt format=engine half=True imgsz=640 device=0
# INT8 icin: int8=True data=<data.yaml>   (kalibrasyon)
```

### YOLO → Hailo .hef (Pi5)  — INT8 zorunlu
```bash
# Hailo Dataflow Compiler (x86 dev makinede; kalibrasyon icin ~100 temsili goruntu):
hailomz compile yolov8n --ckpt yolov8n_pruned.onnx --calib-path calib_imgs/ --hw-arch hailo8l
# (ya da elle: hailo parser onnx -> hailo optimize --calib -> hailo compiler)
# Cikan .hef dosyasini Pi5'e kopyala, REGISTRY'de yolunu ver.
```

### frcnn / rtdetr → ONNX (her iki cihaz için)
```bash
# sunucuda pruned_final.pt'den ONNX uret (pruning script ONNX da kaydedebilir).
# Jetson: onnxruntime-gpu TensorRT EP ONNX'ten engine'i JIT derler (ilk run yavas).
# Pi5: CPU onnxruntime baseline (Hailo two-stage/DETR'i zor derler).
```

---

## 2) Çalıştırma

### Jetson Orin Nano
```bash
# kurulum (cihazda): onnxruntime-gpu, ultralytics, jetson-stats(jtop), psutil, pycocotools
sudo jetson_clocks                 # clock'lari sabitle (tutarli olcum)
sudo nvpmodel -m 0                 # MAXN (veya hedef guc modu)
python3 src/edge_benchmark/benchmark_jetson_orin_nano.py
```

### Raspberry Pi 5 + Hailo-8L
```bash
# kurulum (cihazda): hailort + hailo PCIe driver, onnxruntime, psutil, pycocotools, opencv
python3 src/edge_benchmark/benchmark_rpi5_hailo8l.py
```

---

## Notlar
- `MAP_MAX_IMAGES` ile mAP'i hızlı alt-kümede deneyebilirsin (None = tüm test seti).
- Güç ölçümü cihaz araçlarına bağlı: Jetson `jtop`/`tegrastats`, Pi5 `vcgencmd pmic_read_adc`. Yoksa güç alanları `null` kalır (diğer metrikler etkilenmez).
- **Decode uyarısı**: frcnn/rtdetr ONNX çıktı decode'u ve Hailo on-chip NMS formatı, export/derleme konfigürasyonuna göre değişebilir — ilk çalıştırmada birkaç görüntüde görsel doğrulama yap.
- INT8 (Hailo) ve FP16 (TRT) mAP düşüşü, pruning mAP düşüşünün **üstüne** biner; FP32 sunucu baseline'ı ile karşılaştır.

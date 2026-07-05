# PROJECT.md — `edge_perf` (sonradan gelecek AI/geliştirici için bağlam)

Bu dosya, `edge_perf/` paketini sıfırdan anlaman için yazıldı. Kod yorumları Türkçe;
bu dosya da Türkçe. Önce **Amaç**'ı, sonra **Mimari**'yi, en kritik kısım olan
**Tuzaklar**'ı oku (oradaki her madde gerçek bir debugging turunun sonucu).

---

## 1) Amaç

Termal nesne tespiti projesinde 5 modelin **uç cihaz** performansını karşılaştırmak:
modeller × {base (düz), pruned, quant_fp16, quant_int8} × platform {server, Jetson Orin
Nano, Raspberry Pi 5 + Hailo-8L}. Tek giriş (`main.py`) çalışınca her kombinasyonu ölçer,
JSON/CSV kaydeder ve **sıralar (leaderboard)**. Kullanıcının asıl sorusu: "hangi quantize
(fp16 mı int8 mi)?" → quantize iki satıra bölünmüş (`quant_fp16`, `quant_int8`).

5 model: `yolov8n`, `yolo_fir_v5s` (YOLOv5s tabanlı), `faster_rcnn_resnet50`,
`faster_rcnn_mobilenet`, `rtdetr_v2_r18vd`. 3 sınıf: Person / Car / OtherVehicle.

## 2) Donanım gerçeği (ÖNEMLİ)

Geliştirme **H200 MIG sunucu**da yapıldı; gerçek Jetson/Hailo latency+güç ancak o
cihazlarda alınır. Bu yüzden `main.py` **platformu otomatik algılar** ve her platformda
farklı backend koşar. Sunucuda `torch` baseline çalışır (doğrulama + referans). Jetson
(TensorRT) ve Pi (HailoRT/onnxruntime-CPU) yolları cihazda koşar.

## 3) Mimari (dosya dosya)

```
edge_perf/
  main.py              # TEK giriş: platform algıla -> (model×variant) matrisi -> bench -> kaydet+sırala
  config.py            # TEK kayıt defteri: MODELS listesi, PLATFORM_VARIANTS matrisi, yol çözümleyiciler
  bench_core.py        # cihaz-bağımsız çekirdek: latency_stats, BackgroundSampler (güç), evaluate_map (COCO),
                       #   derive_efficiency, rank_models/composite_score, save_results (+leaderboard)
  backends.py          # detect_platform() + tüm backend sınıfları (aşağıda)
  prepare_models.py    # base/pruned .pt -> models/<key>/{base,pruned}.pt (kopya veya --link)
  convert/
    export_onnx.py     # .pt -> fp32 .onnx (yolo: ultralytics; frcnn/rtdetr: torch.onnx)
    quantize_onnx.py   # fp32 .onnx -> _fp16.onnx (onnxconverter_common) + _int8.onnx (quantize_dynamic)
    build_tensorrt.py  # JETSON'da: YOLO .pt -> _<prec>.engine (fp16/int8)
    build_hailo.sh     # x86 Hailo DFC: YOLO .onnx -> .hef (int8, kalibrasyonlu)
  models/<key>/        # tüm artifact'lar (gitignore'lu); prepare+convert üretir
  results/             # JSON/CSV/ranking çıktıları (gitignore'lu)
  requirements_rpi5.txt
  README.md            # kullanıcı kılavuzu
  PROJECT.md           # bu dosya
```

### Backend sınıfları (`backends.py`) — hepsi aynı arayüz
`infer(image_bgr) -> (boxes_xyxy_pixel, scores, class_idx_0based)`, `class_names`, `close()`.

- **ServerTorchBackend**: CUDA/CPU torch. yolo→ultralytics; frcnn→`model_obj` (pruned) veya
  `load_faster_rcnn` (base); rtdetr→HF model + `AutoImageProcessor` (post_process). fp32/fp16.
- **TrtYoloBackend**: ultralytics `YOLO(artifact)` — `.engine`/`.onnx`/`.pt` hepsini açar
  (Jetson engine VE Pi onnx-CPU için ortak kullanılır).
- **TrtOnnxBackend**: onnxruntime TensorRT EP (Jetson, frcnn/rtdetr `.onnx`).
- **HailoBackend**: HailoRT `.hef` (Pi NPU, int8, on-chip NMS decode).
- **CpuOnnxBackend**: onnxruntime CPU (Pi, frcnn/rtdetr `.onnx`; fp32/fp16/int8 aynı API).

### Akış kontrolü
`config.PLATFORM_VARIANTS[platform]` = `[(variant, structure, precision, backend_kind), ...]`.
`main.build_backend()` bunu artifact'a + backend sınıfına çevirir; artifact yoksa **SKIP**.
`config.resolve_*` fonksiyonları tek yol-otoritesidir (başka yerde sabit yol yok).

## 4) Doğrulama durumu

- ✅ **Server torch**: 5 modelin de base/pruned/quant_fp16 ölçümü uçtan uca çalışıyor
  (latency, mAP, params, disk, ranking). quant_int8 server'da yok → temiz SKIP.
- ✅ ONNX üretimi: yolo + rtdetr için fp32/fp16/int8 üretildi.
- ⚠️ frcnn ONNX export **başarısız** (torchvision two-stage; "NoneType.shape"). RPi'de frcnn
  → **torch-CPU yedeği** (`main.build_backend` rpi dalı `.pt`'den fp32).
- ⛔ onnx-CPU runtime **bu sunucuda doğrulanamadı**: `onnxruntime-gpu 1.26` `libcudart.so.13`
  ister, sistemde CUDA 12 var → import çöküyor. **Pi'de ARM CPU wheel ile sorun yok.** Decode
  mantığı, doğrulanmış torch rtdetr decode'u ve `src/edge_benchmark` ile birebir aynı.
- ⛔ Jetson `.engine` / Hailo `.hef`: cihaz/toolchain gerektirir; burada üretilemedi.

## 5) TUZAKLAR (her biri gerçek bir hata turu — tekrar düşme)

1. **rtdetr imgsz = 480**, 640 DEĞİL. Eğitim `IMAGE_SIZE=480` (train_rtdetr.py). 640 verince
   güven ~0.03'e düşer, mAP=0. `AutoImageProcessor`'a `size={"width":480,"height":480}` ver.
2. **conf eşiği DETR ailesinde ~0.001 olmalı.** RT-DETR/DETR düşük-kalibre skor üretir
   (max ~0.05). Yüksek eşik (0.3) tüm tahminleri siler → mAP=0. Projenin kendi MAPEvaluator'ı
   da `threshold=0.0` kullanıyor. `--conf` varsayılanı 0.001.
3. **mAP sınıf eşlemesi POZİSYON-tabanlı** (`evaluate_map(model_class_names=None)`). YOLO
   `model.names` bu projede gt ile TUTARSIZ (isimle eşleme YANLIŞ olur). frcnn/rtdetr zaten
   Person/Car/OtherVehicle sırasında.
4. **frcnn yükleme: pruned ≠ base.** pruned ckpt `model_obj` (hazır nn.Module) tutar
   (rebuild edilemez); base ckpt `model` (state_dict) tutar → `load_faster_rcnn` ile
   `create_faster_rcnn`+load. Otomatik ayrım: ckpt'te `model_obj` var mı?
5. **frcnn labels 1-based** (0=arka plan) → mAP'te `-1` yap. rtdetr/yolo 0-based.
6. **Pruned ağırlıklar finetune'suz** (`prune_*` wrapper'larında `DO_FINETUNE=False`):
   - `yolo_fir_v5s` finetune'lu pruned = `runs_pruing/finetune/weights/best.pt` (args.yaml
     doğrular: model=yolo_fir, epochs=10). `pruned_final.pt` ise KIRIK (mAP 0.569→0.0005).
   - `rtdetr` pruned_final.pt finetune'suz → mAP≈0. Anlamlı sonuç için finetune şart.
   - `yolov8n` pruned çıktısı YOK (`runs_pruing/yolov8n` boş). config bilerek paylaşımlı
     `runs_pruing/finetune`'u yolov8n'e fallback ETMEZ (o yolo_fir'in) → yolov8n pruned SKIP.
   - frcnn pruned (backbone-only hafif) finetune'suz da makul.
   → Düzeltmek için: `src/Pruning/prune_*` içinde `DO_FINETUNE=True` ile yeniden koş.
7. **datasets/ namespace clash**: PROJECT_ROOT'u sys.path'e EKLEME (yerel `datasets/` HF
   `datasets`'i gölgeler). Sadece `models/faster_rcnn` gibi alt-dizinleri ekle.
8. **onnxruntime-gpu çakışması (server)**: torch (cu124) ile onnxruntime-gpu 1.26 (cuda13)
   aynı süreçte çakışır. Sunucuda onnx-CPU testi yapma; Pi'de ARM CPU wheel kullan.
9. **MP FD tükenmesi** (bu host): DataLoader worker'ları EMFILE veriyor → ağır iş yaparsan
   `torch.multiprocessing.set_sharing_strategy("file_system")`.
10. **Hailo int8 zorunlu**: Hailo-8L sadece int8 çalışır; `.hef` ancak x86 + Hailo DFC ile
    kalibrasyon görüntüleriyle derlenir. frcnn/rtdetr Hailo'da derlenmez → Pi CPU'ya düşer.

## 6) Genişletme

- **Yeni model**: `config.MODELS`'e giriş ekle (key, type, imgsz, normal_pt/pruned_pt adayları).
  Backend'ler type'a göre (yolo/frcnn/rtdetr) çalışır; yeni tip eklersen `ServerTorchBackend`
  + decode + (varsa) onnx decode yaz.
- **Yeni platform/precision**: `PLATFORM_VARIANTS` + `resolve_device_artifact` + `build_backend`.
- **Sıralama metriği**: `bench_core.RANK_METRICS` / `composite_score` ağırlıkları.

## 7) İlgili proje parçaları

- `src/edge_benchmark/` — bu paketin atası (sadece pruned, tek main yok). Kanıtlanmış
  Jetson/Hailo decode mantığı buradan uyarlandı.
- `src/Pruning/` — 5 model pruning pipeline (prune_lib_{yolo,frcnn,rtdetr}). Pruned ağırlık
  kaynağı; `DO_FINETUNE` ayarı burada.
- `models/faster_rcnn/model_loaders.py` (`load_faster_rcnn`), `models/rtdetr_v2_r18vd/`
  (train + MAPEvaluator) — yükleme/eval referansı.
- Test seti: `dataset/2x_augmented_coco_dataset/dataset_augmented/test/` (6097 görüntü).

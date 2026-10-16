# Uç Cihazlar İçin Optimize Edilmiş Nesne Tespit Modelleri

Termal ve RGB havadan görüntülerde insan/araç tespiti yapan derin öğrenme modellerinin
**karşılaştırmalı eğitimi, budanması, kuantizasyonu ve uç cihaz ölçümü**.

Savunma Sanayii Başkanlığı **SAYZEK ATP** programı kapsamında, 70 kişilik program bünyesindeki
nesne tespiti ekibinde yürütülmüştür.

> **Veri:** Eğitimde kullanılan görüntüler kamuya açık veri setlerinden
> ([HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset),
> [RGBT-Tiny](https://github.com/XinyiYing/RGBT-Tiny)) türetilmiş ve birleştirilmiştir.
> Veri setleri ve model ağırlıkları bu depoda yer almaz.

---

## Ne yapıldı

**1. Mimari karşılaştırması.** Tek aşamalı (YOLOv8n, YOLO11n, YOLO26n, YOLO-FIR v5s, SSD),
iki aşamalı (Faster R-CNN — ResNet50 / MobileNetV3 / DLA34) ve transformer tabanlı
(RT-DETR v2 R18vd, DETR-ResNet50) dedektörler aynı veri setinde eğitilip karşılaştırıldı.

**2. Sınıf dengesizliği.** Havadan görüntülerde nesne sınıfları çok dengesiz dağıldığı için
**Focal Loss** (Lin ve ark. 2017) ve **Class-Balanced Loss** (Cui ve ark. 2019, effective number
of samples) uygulandı; beta değerleri üzerinde sweep yapılarak taban modelle karşılaştırıldı.

**3. Budama (structured pruning).** Benchmark'ta öne çıkan ve birbirinden farklı **5 model**,
`torch_pruning` (DepGraph) ile kanal düzeyinde fiziksel olarak budandı, ardından yeniden eğitildi.

**4. Uç cihaza taşıma.** ONNX dönüşümü, INT8 dinamik kuantizasyon ve TensorRT motor üretimi
(YOLO için doğrudan `.engine`, Faster R-CNN / RT-DETR için ONNX Runtime TensorRT EP + motor
önbellekleme).

**5. Uç cihaz ölçümü.** NVIDIA Jetson Orin ve Raspberry Pi 5 + Hailo-8L üzerinde gecikme,
FPS, **güç ve enerji/kare (mJ)**, FPS/Watt, tepe RAM, sıcaklık ve kuantizasyon sonrası mAP.

---

## Sonuçlar

### Sınıf dengesizliği stratejilerinin karşılaştırması

| Model | Kayıp | Epoch | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| **YOLOv11n** | default | 68 | **0.8134** | **0.5743** | 0.7603 | 0.8011 |
| YOLOv8n | Class-Balanced | 111 | 0.8120 | 0.5687 | 0.7872 | **0.8399** |
| YOLOv8n | Class-Balanced | 129 | 0.8015 | 0.5664 | 0.7697 | 0.7815 |
| YOLOv11n | Focal | 53 | 0.7927 | 0.5080 | 0.7991 | 0.7826 |
| YOLOv11n | default (baseline) | 50 | 0.7742 | 0.5035 | 0.7648 | 0.7971 |

Class-Balanced Loss recall'ı belirgin biçimde yükseltiyor (0.84), ancak mAP@50'de taban modeli
geçmiyor — havadan küçük nesne tespitinde kayıp fonksiyonu değişikliğinin sınırlı kaldığı görüldü.

### Uç cihaz ölçümü (Jetson Orin)

| Model | mAP@50 | FPS | Çıkarım | Parametre |
|---|---|---|---|---|
| YOLOv8n (CB) | 0.8919 | ~40 | ~25 ms | ~3.2 M |
| Faster R-CNN ResNet50 | — | ~3.9 | ~254 ms | ~19 M |

İki aşamalı dedektörlerin uç cihazda kullanılamayacağı ölçümle gösterildi; dağıtım için
tek aşamalı hafif modellerde karar kılındı.

---

## Depo yapısı

```
src/
├── focal_loss_func.py        # Focal Loss (Lin ve ark. 2017)
├── cb_loss.py                # Class-Balanced Loss (Cui ve ark. 2019) — sınıflandırma + tespit
├── Pruning/                  # torch_pruning / DepGraph ile 5 modelin budanması
├── Quantization/             # ONNX export + INT8 dinamik kuantizasyon
├── edge_benchmark/           # Jetson Orin Nano ve RPi5 + Hailo-8L ölçüm hattı
├── model_benchmark/          # hız, bellek, doğruluk ve determinizm ölçümü
├── benchmark/                # YOLO ve U-Net cihaz üstü benchmark script'leri
└── ipynb_files/              # DETR ve Faster R-CNN eğitim defterleri

edge_perf/convert/            # .pt → ONNX → TensorRT/Hailo dönüşüm hattı
models/                       # mimari bazlı eğitim defterleri ve çıktıları
runs/ runs_new/ runs_pruing/  # eğitim koşuları, eğriler, karışıklık matrisleri
results/                      # örnek çıkarım görselleri
bechmarks/                    # uç cihaz ölçüm çıktıları (JSON)
```

Ölçüm tabloları: `model_skor_tablosu.xlsx`, `training_results_2.xlsx`
Budama özeti: [`src/Pruning/PRUNING_OZET.md`](src/Pruning/PRUNING_OZET.md)
Ölçüm hattı: [`src/edge_benchmark/README.md`](src/edge_benchmark/README.md)

## Katkıda bulunanlar

Proje ortak bir geliştirme sunucusunda yürütüldü ve commit'lerin büyük bölümü sunucuda oturum
açık olan tek bir hesap üzerinden atıldı. Bu nedenle git geçmişi iş bölümünü doğru yansıtmıyor.
Gerçek katkı dağılımı:

- **Uğurhan Daşdemir** — benchmark altyapısı, model karşılaştırma ve ölçüm hattı, budama
  script'leri, YOLO-FIR / termal model eğitimleri, uç cihaz ölçümlerinin yürütülmesi
- **Erhan Yüksel Turgut** — DETR ve Faster R-CNN eğitim hatları, Focal Loss ve Class-Balanced
  Loss uygulamaları ve sweep'leri, YOLO ve RT-DETR modellerinin ONNX dönüşümü ve fp16/int8
  sürümlerinin üretimine katkı
- **Hakan Enes Demir** — model ablasyon denemeleri ve eğitim koşularının karşılaştırılması

## Kaynaklar

- Lin ve ark. (2017), *Focal Loss for Dense Object Detection* — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Cui ve ark. (2019), *Class-Balanced Loss Based on Effective Number of Samples* — [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
- Fang ve ark. (2023), *DepGraph: Towards Any Structural Pruning* — [arXiv:2301.12900](https://arxiv.org/abs/2301.12900)

# faster_cnn3 — v4 Düzeltmeleri

Kötü test sonuçlarının (Person P=0.08, Car P=0.20, massive FP) kök nedenlerini
giderir. Test confusion matrix'inde epoch 115 modeli `bg → Person: 48915`,
`bg → Car: 36347` gibi devasa false positive üretiyordu. Aşağıdaki üç hatanın
birleşimi bunu açıklıyor.

---

## Hata 1 — Tutarsız kanal işleme (KRİTİK, kök neden)

**Dosya:** `object-detection-fasterrcnn.ipynb`, cell-7 (`ThermalDetection._load_image`)

**Problem:** Dataset karışık 3 format içeriyor (2000 örnek üzerinden ölçüm):

| Format | Oran |
|---|---|
| Gerçek 1-kanal gri | %25 |
| 3-kanal, R=G=B (gri olarak kayıtlı) | %67 |
| Gerçek renkli (pseudo-color termal / RGB) | %8 |

Eski kod `cv2.imread(path)` kullanıyordu (default `IMREAD_COLOR`). Bu:
- 1-kanal dosyayı → 3 aynı kanala kopyalar (R=G=B)
- 3-kanal renkli dosyayı → gerçek BGR olarak okur; `BGR2RGB` swap eder

Sonuç: model **iki farklı dağılım** görüyor — %92 efektif-gri (R=G=B) + %8 gerçek
renkli. ImageNet pretrained backbone bu distribution shift'e çok duyarlı;
test setinde aşırı FP'nin birincil nedeni.

**Düzeltme:** Hepsini `IMREAD_GRAYSCALE` ile oku, sonra 3 özdeş kanala genişlet.

```python
img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(full)
return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

Termal içerik yalnızca intensity olduğu için %8 renkli örneklerde gerçek bilgi
kaybı yok. Tüm dataset tek dağılıma (R=G=B gri) zorlanıyor.

---

## Hata 2 — Etiketler dönüşümden önceki target'tan okunuyor

**Dosya:** `object-detection-fasterrcnn.ipynb`, cell-7 (`ThermalDetection.__getitem__`)

**Problem:** Eski kod `boxes`'u `transformed['bboxes']`'ten alırken `labels`'ı
ham `target`'tan okuyordu:

```python
boxes = transformed['bboxes']
...
targ['labels'] = torch.tensor([t['category_id'] + 1 for t in target], ...)  # HAM!
```

Mevcut transform zincirinde (`Resize` + `HorizontalFlip` + `clip=True`)
albumentations kutu düşürmediği için uyumsuzluk tetiklenmiyor; doğrulandı.
Ama ileride `min_area`, `min_visibility`, `RandomCrop` eklendiğinde sessizce
yanlış label'lı kutularla eğitim yapılacaktı.

**Düzeltme:** `category_id` zaten `boxes[i][4]`'te korunuyor (çünkü `boxes`
`[x, y, w, h, cls]` formatında albumentations'a gönderiliyor). Oradan oku:

```python
new_boxes, new_labels = [], []
for box in boxes:
    xmin, ymin = box[0], box[1]
    xmax, ymax = xmin + box[2], ymin + box[3]
    new_boxes.append([xmin, ymin, xmax, ymax])
    new_labels.append(int(box[4]) + 1)
```

`iscrowd` da `len(new_labels)` uzunluğunda sıfır tensörüne sabitlendi (bu
dataset'te crowd yok; boy eşlemesi güvence).

---

## Hata 3 — `evaluate()` BatchNorm stats'ını val verisiyle kirletiyor

**Dosya:** `object-detection-fasterrcnn.ipynb`, cell-29 (`evaluate`)

**Problem:** Faster R-CNN yalnızca `train()` modundayken loss dict döndürür.
Eski kod val loss için `model.train()` çağırıyor, ama bu **tüm BatchNorm
katmanlarını da train moduna alıyor** → val batch'leri her epoch'ta
`running_mean`/`running_var`'ı güncelliyor. 115 epoch boyunca BN stats'ı
yavaş yavaş val dağılımına doğru kayıyor; inference kalitesini düşürüyor.

**Düzeltme:** Modeli `train()` moduna al, ardından sadece BN katmanlarını
manuel olarak `eval()`'e geri al:

```python
model.train()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
```

Bu şekilde model loss dict dönmeye devam eder ama BN istatistikleri
dondurulur.

---

## Hata 4 — CB Loss beta aşırı yüksek, ağırlıklar düzleşiyor

**Dosya:** `object-detection-fasterrcnn.ipynb`, cell-25

**Problem:** `beta=0.9999` + büyük class counts (`[2.8M, 335K, 577K, 36K]`)
ile `effective_num = (1-β^n)/(1-β)` formülü tüm sınıflarda ~10000'e
saturate oluyor. Normalizasyon sonrası weights ≈ `[1.0, 1.0, 1.0, 1.03]`
— pratikte sınıf dengeleme **yok**, CB Focal standart Focal'a çöküyor.
Oysa dataset'te 36K OtherVehicle vs 577K Car arasında 16× fark var.

**Düzeltme:** `beta=0.999`. Bu değerde:
- 36K → effective_num ≈ 972
- 577K → effective_num ≈ 1000
- weights arası belirgin fark → az örnekli sınıflar gerçekten yukarı çekilir.

---

## Beklenen etki

- **Hata 1** tek başına test precision'ını büyük ölçüde artırmalı (distribution
  shift ortadan kalkınca model bg'yi reddetmeyi doğru öğrenir).
- **Hata 3** çok epoch'lu eğitimde eval ve test arası tutarlılığı artırır.
- **Hata 4** OtherVehicle için zaten iyi olan recall'u daha da iyileştirir ve
  Person/Car arasındaki dengeyi dokunmadan bırakır.
- **Hata 2** şu anda sonuca etki etmiyor ama gelecek augmentation'lara karşı
  güvence.

## Eğitim protokolü (öneri)

Fix'ler image loading'i değiştirdiği için eski `last.pt`/`best.pt`'in BN stats'ı
eski (kirli + karışık kanal) dağılıma göre kalibre. **Sıfırdan eğitim önerilir:**

```bash
rm /home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs/faster_cnn3/last.pt
rm /home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/runs/faster_cnn3/best.pt
```

Sonra notebook'u baştan (cell-0'dan itibaren) çalıştır.

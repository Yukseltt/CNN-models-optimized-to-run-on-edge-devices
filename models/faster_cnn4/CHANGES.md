# Faster R-CNN — Sürüm Değişiklikleri

Bu dosya `faster-cnn` (v1) → `faster-cnn2` (v2) → `faster_cnn3` (v3) → `faster_cnn4` (v4)
arasında notebook'ta yapılan tüm değişiklikleri listeler.

Önceki sürüm geçmişi için bkz. [faster_cnn3/CHANGES.md](../faster_cnn3/CHANGES.md).

---

## v3 → v4 (faster_cnn3 → faster_cnn4)

v3'ün eğitim kaydında şunlar gözlemlendi:
- `Accuracy: 0.3252`, `Macro Precision: 0.2128`, `Macro Recall: 0.4390`
- `Person P:0.0815` → model kişileri büyük çoğunlukla atlıyor (FN:11914)
- `background` satırında 48915 Person + 36347 Car FP → aşırı false positive
- v2/v3'te `beta=0.9999` ile CB Loss sınıflar arası ağırlık farkı ~%3 — CB Loss fiilen pasif
- Grayscale/renkli görüntü karışımı modelin farklı dağılımlar öğrenmesine neden oluyordu

### Cell 7 — `_load_image` (**Kritik düzeltme**)

- **`cv2.imread()` (default BGR) → `cv2.imread(IMREAD_GRAYSCALE)` + `cv2.COLOR_GRAY2RGB`.**
- **Neden:** Dataset'te 1-kanal gri, 3-kanal (R=G=B) ve ~%8 gerçek renkli
  (pseudo-color termal) görüntüler **karışık** bulunuyor. `IMREAD_COLOR` (default):
  - Gri görüntüleri 3 özdeş kanala kopyalıyor (doğru).
  - Renkli dosyaları BGR olarak okuyor → bu görüntülerde kanallar farklı.
  - Sonuç: model iki farklı dağılım görüyor (tüm kanallar eşit vs kanallar farklı) →
    distribution shift → test setinde aşırı FP.
- **Çözüm:** Hepsini `IMREAD_GRAYSCALE` ile oku, sonra 3 özdeş kanala genişlet.
  Termal içeriği intensity/sıcaklık bilgisi taşıdığından renkli ~%8'de anlamlı bilgi kaybı yok.

```python
img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
```

### Cell 7 — `__getitem__` label extraction (**Güvenlik düzeltmesi**)

- **Etiketler artık dönüşüm SONRASI kutulardan okunuyor (`box[4]`), raw `target`'tan değil.**
- **Neden:** Eski kod dönüşümden önce `target`'tan etiket alıyordu; eğer albumentations
  bir kutuyu `min_area` veya `RandomCrop` ile düşürürse kutu/etiket sayısı uyumsuz kalır →
  sessiz, tespit edilmesi güç yanlış eğitim. Şu andaki transform'larda (Resize+HFlip,
  `clip=True`) kutu düşmüyor, ancak ileride genişletildiğinde bu bug tetiklenir.

```python
for box in boxes:   # boxes = transformed bboxes
    new_labels.append(int(box[4]) + 1)   # box[4] = category_id (dönüşüm sonrası)
```

### Cell 25 — CB Loss `beta` (**Kritik düzeltme**)

- **`beta=0.9999` → `beta=0.999`.**
- **Neden:** `beta=0.9999` + büyük `class_counts` (~300K–2.8M) ile
  `effective_num = (1 - beta^n) / (1 - beta)` formülü tüm sınıflarda ~10000'e
  saturate oluyor → normalizasyon sonrası ağırlıklar ~[1.0, 1.0, 1.0, 1.03] →
  pratikte sıradan focal loss, **sınıf dengeleme YOK**.
  Datasette 36K OtherVehicle vs 577K Car gerçek bir dengesizlik var.
  `beta=0.999` ile: 36K → effective_num ~972, 577K → ~1000 → belirgin ağırlık
  farkı oluşur, OtherVehicle ve Person gerçekten yükseltilir.

```python
cb_cls_loss_fn = ClassBalancedDetectionClsLoss(
    class_counts=class_counts_with_bg,
    beta=0.999,   # 0.9999 -> 0.999
    gamma=2.0, loss_type="focal", background_idx=None, device=device,
)
```

### Cell 29 — `evaluate`: BatchNorm eval mode (**Doğruluk düzeltmesi**)

- **Val loss hesaplanırken BatchNorm katmanları `eval()` moduna alınıyor.**
- **Neden:** Faster R-CNN loss döndürmek için `train()` moduna ihtiyaç duyar.
  Ancak `model.train()` BatchNorm'un `running_mean` / `running_var` istatistiklerini
  val verisi ile günceller → 150 epoch boyunca BN stats val'den kirlenir →
  hem eğitim istatistikleri bozulur hem de sonraki epoch'larda train ve val dağılımı
  yapay olarak yakınlaşır.
- **Çözüm:** `model.train()` sonrası tüm `_BatchNorm` katmanlarını `m.eval()` ile
  geri al. Böylece loss döndürülür ama BN running stats güncellenmez.

```python
model.train()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
```

### Cell 24, 36, 37 — Yol güncellemeleri

- `save_dir`, `xlsx_path`, `out_dir`, `best_ckpt` → tümü `faster_cnn3` → `faster_cnn4`.

---

## Özet: her sürümde aslında ne değişti

| Alan                     | v1                    | v2                          | v3                                  | v4                                        |
|--------------------------|-----------------------|-----------------------------|-------------------------------------|-------------------------------------------|
| LR (head)                | 0.01                  | 0.005                       | 0.005                               | 0.005                                     |
| LR (backbone)            | 0.01 (tek grup)       | 0.005 (tek grup)            | **0.0005 (ayrı param group)**       | 0.0005                                    |
| Warmup                   | yok                   | 1000 iter lineer            | aynı                                | aynı                                      |
| Grad clip                | 5.0                   | 1.0                         | 1.0                                 | 1.0                                       |
| Backbone freeze          | yok                   | ilk 5 epoch frozen          | aynı                                | aynı                                      |
| Best checkpoint kriteri  | total loss            | mAP50-95                    | mAP50-95                            | mAP50-95                                  |
| Early stopping           | yok                   | patience=15                 | aynı                                | aynı                                      |
| CB Loss bg handling      | `background_idx=0` (**BUG**) | `background_idx=0` (**BUG**) | **`background_idx=None` + bg count** | aynı                                 |
| CB Loss beta             | 0.9999                | 0.9999                      | 0.9999                              | **0.999 (gerçek sınıf dengeleme)**        |
| Görüntü okuma            | BGR (karışık dağılım) | BGR                         | BGR                                 | **GRAYSCALE→RGB (distribution shift YOK)**|
| Label extraction         | raw target (dönüşüm öncesi) | raw target            | raw target                          | **dönüşüm sonrası (box[4])**              |
| Val BN stats kirlenmesi  | var                   | var                         | var                                 | **YOK (BatchNorm.eval() koruma)**         |
| targets cast             | `torch.tensor`        | `torch.tensor`              | **`torch.as_tensor`**               | aynı                                      |
| Yollar                   | `faster_cnn`          | tutarsız                    | `faster_cnn3`                       | **`faster_cnn4`**                         |

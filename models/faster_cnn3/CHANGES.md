# Faster R-CNN — Sürüm Değişiklikleri

Bu dosya `faster-cnn` (v1) → `faster-cnn2` (v2) → `faster_cnn3` (v3)
arasında notebook'ta yapılan tüm değişiklikleri tek tek listeler.

---

## v1 → v2 (faster-cnn → faster-cnn2)

v1'in ilk eğitiminde gözlemlenen sorunlar: `loss_box` 0.089 → 0.30'a
çıktı, en iyi `best.pt` ilk epoch'ta kaydedilip bir daha güncellenmedi,
son ~34 epoch plato'da boşa harcandı. v2 bunlara karşılık şunları getirdi.

### Cell 15 — Model kurulumu
- **Eklendi:** Backbone başlangıçta donduruluyor (ilk 5 epoch boyunca
  sadece yeni head eğitilir).
- **Neden:** Yeni eklenen `box_predictor` rastgele ağırlıkla başlıyor ve
  ilk epoch'larda büyük gradientler üreterek ImageNet ön-eğitilmiş
  backbone'u bozuyordu. `loss_box`'un 0.089 → 0.30 fırlaması bu
  instability'nin işaretiydi.

```python
for p in model.backbone.parameters():
    p.requires_grad = False
```

### Cell 24 — Optimizer & LR
- **LR 0.01 → 0.005** düşürüldü.
  - **Neden:** Faster R-CNN + SGD+Momentum için güvenli aralık 0.002–0.005;
    0.01'de box regresyon kafası salınım yapıp yakınsayamıyordu.
- **Optimizer'a tüm parametreler veriliyor** (frozen dahil):
  `params = list(model.parameters())`.
  - **Neden:** Cell 30'da backbone açıldığında aynı optimizer state'iyle
    devam edebilmek için.

### Cell 28 — Training loop
- **Warmup eklendi** (ilk 1000 iterasyon, 5e-6 → 0.005 lineer).
  - **Neden:** Head rastgele başlıyor; warmup olmadan gradient patlaması.
- **Gradient clipping `max_norm` 5.0 → 1.0** sıkılaştırıldı.
  - **Neden:** v1'de clip=5.0 box regresyon salınımını engelleyemedi.
    1.0 standart güvenli değer, özellikle MobileNet gibi küçük backbone'larda.

### Cell 30 — Training orchestration
- **`best.pt` kriteri loss → mAP50-95** değiştirildi.
  - **Neden:** v1'de total loss bir daha ilk epoch değerinin altına inmediği
    için `best.pt` yalnızca epoch 1'de kaydedildi. Detection'da model
    seçiminin doğru kriteri mAP'tir, loss değil.
- **Early stopping (patience=15)** eklendi.
  - **Neden:** v1'de son 34 epoch (5+ saat) plato'da boşa harcandı.
- **Unfreeze logic:** `epoch == UNFREEZE_EPOCH (=5)` olduğunda
  `model.backbone.parameters()` için `requires_grad=True`.
- **Resume:** Checkpoint'tan devam ederken, eğer `start_epoch >= UNFREEZE_EPOCH`
  ise backbone hemen açılıyor; `best_map` ve `epochs_no_improve` da
  checkpoint'tan geri yükleniyor.

### Cell 36, 37 — Plot & test
- Çıktı dizinleri ve xlsx yolları `faster_cnn` → `faster_cnn2` olarak
  güncellendi (kısmen — Cell 24 hedefi zaten `faster_cnn3` yazıyordu,
  v3'te düzeltildi; bkz. aşağıda).

---

## v2 → v3 (faster-cnn2 → faster_cnn3)

v2'nin 57 epoch'luk eğitim kaydında şunlar gözlemlendi:
- `loss_classifier` anormal düşük (0.0017)
- `Precision: 0.12`, `Recall: 0.31` → model her şeyi foreground işaretliyor
- `mAP50-95: 0.099` civarında plato
- Backbone açıldıktan sonra `loss_box` 0.03 → 0.26'ya (10×) **çıktı**

Kök neden tespit edildi: Class-Balanced Loss yanlış yapılandırılmış ve
optimizer tek bir LR ile backbone + head'i eşit hızda güncellemeye
zorluyor. v3 bunları düzeltir.

### Cell 24 — Optimizer: param groups
- **Tek LR → iki param group.** Backbone LR head LR'nin 1/10'u.
  - `head_params` → `lr=0.005`
  - `backbone_params` → `lr=0.0005`
- **Neden:** v2'de backbone epoch 5'te açıldığında `loss_box` 0.03 → 0.26'ya
  patladı — bu klasik fine-tune instabilitesidir. Pretrained backbone'u
  rastgele init head ile aynı LR'de güncellemek backbone'daki ImageNet
  özelliklerini bozuyor. Param groups ile backbone frozen iken (epoch 0-4)
  düşük LR hiçbir şey yapmaz, unfreeze sonrası yumuşak bir fine-tune sağlar.
- **`eta_min` 1e-5 → 1e-6:** CosineAnnealingLR param group'larla orantılı
  çalıştığı için backbone LR'nin de yeterince aşağı inebilmesi için.
- **Yol düzeltmesi:** `save_dir` standart olarak `runs/faster_cnn3`. (v2'de
  Cell 24 zaten `faster_cnn3` yazıyordu ama Cell 36 ve 37 hâlâ
  `faster_cnn2`'den okuyordu — bu tutarsızlık giderildi.)

```python
backbone_params = list(model.backbone.parameters())
backbone_ids = {id(p) for p in backbone_params}
head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

optimizer = torch.optim.SGD(
    [
        {"params": head_params,     "lr": 0.005},
        {"params": backbone_params, "lr": 0.0005},
    ],
    momentum=0.9, nesterov=True, weight_decay=1e-4,
)
```

### Cell 25 — CB Loss (**KRİTİK düzeltme**)
- **`background_idx=0` → `background_idx=None`.**
- **`class_counts` listesinin başına `bg_count = 3 * sum(fg)` eklendi.**
- **Neden:** v1 ve v2'deki yapılandırma CB loss'un forward'ında
  background ROI'lerini **tamamen** loss'tan düşürüyordu
  ([cb_loss.py:476-480](../../src/cb_loss.py#L476-L480)). Faster R-CNN'in
  `fg_bg_sampler`'ı her batch'te ~%75 bg ROI üretir; bunlar loss'a
  girmeyince model "bu arka plan" sinyalini **hiç almıyor** → false
  positive'leri reddetmeyi öğrenemiyor. Metriklerdeki tablo birebir bunu
  yansıtıyordu:
  - `loss_classifier: 0.0017` (absürt düşük, sahte)
  - `P=0.12, R=0.31` → her şeyi foreground işaretleme
  - `loss_box` patlaması → cls sinyali bozuk, box head dengesiz gradient alıyor
- **Çözüm (Option A):** `background_idx=None` kullan, class_counts
  listesinin başına `bg_count` ekle. Bu sayede bg sınıfı CB ağırlık vektöründe
  yer alır (formül bg'yi "çok sık" sınıf olarak görüp düşük ağırlık verir)
  ama forward'da **dışlanmaz** — bg ROI'ler cls loss'a katkıda bulunur.
- **`bg_count = 3 * sum(fg)`:** Faster R-CNN sampler'ın bg/fg oranıyla
  (~3:1) uyumlu seçildi.

```python
bg_count = 3 * sum(class_counts_cb)
class_counts_with_bg = [bg_count] + list(class_counts_cb)
cb_cls_loss_fn = ClassBalancedDetectionClsLoss(
    class_counts=class_counts_with_bg,
    beta=0.9999, gamma=2.0, loss_type="focal",
    background_idx=None,   # bg loss'a girsin
    device=device,
)
```

### Cell 28 — Training loop
- **`torch.tensor(v)` → `torch.as_tensor(v)`** (targets conversion).
- **Neden:** `v` zaten `Tensor` ise `torch.tensor(v)` hem `UserWarning`
  fırlatıyor hem de gereksiz kopya yapıyor. `torch.as_tensor` sıfır-kopyalı
  dönüşümdür.

### Cell 36 — Plot
- `xlsx_path` ve `out_dir` `faster_cnn2` → `faster_cnn3` güncellendi.
  (v2'de inconsistent kalmıştı.)

### Cell 37 — Test değerlendirme
- **Yollar** `faster_cnn2` → `faster_cnn3`.
- **`ckpt.get('loss', float('nan'))` → `ckpt.get('map50_95', float('nan'))`.**
- **Neden:** v2'de Cell 30 `best.pt`'yi mAP50-95 ile kaydetmeye başladı
  ama `loss` key'i artık yazılmıyordu. Eski kod her zaman `nan` basıyordu.

---

## Özet: her sürümde aslında ne değişti

| Alan                     | v1                    | v2                          | v3                                  |
|--------------------------|-----------------------|-----------------------------|-------------------------------------|
| LR (head)                | 0.01                  | 0.005                       | 0.005                               |
| LR (backbone)            | 0.01 (tek grup)       | 0.005 (tek grup)            | **0.0005 (ayrı param group)**        |
| Warmup                   | yok                   | 1000 iter lineer            | aynı                                |
| Grad clip                | 5.0                   | 1.0                         | 1.0                                 |
| Backbone freeze          | yok                   | ilk 5 epoch frozen          | aynı                                |
| Best checkpoint kriteri  | total loss            | mAP50-95                    | mAP50-95                            |
| Early stopping           | yok                   | patience=15                 | aynı                                |
| CB Loss bg handling      | `background_idx=0` (**BUG**) | `background_idx=0` (**BUG**) | **`background_idx=None` + bg sınıf sayısı** |
| targets cast             | `torch.tensor`        | `torch.tensor`              | **`torch.as_tensor`**                |
| Yollar                   | `faster_cnn`          | tutarsız (`faster_cnn2` / `faster_cnn3`) | standart: `faster_cnn3`       |
| best.pt test print       | `loss` okuyor         | `loss` okuyor (**nan basıyor**) | **`map50_95` okuyor**                |

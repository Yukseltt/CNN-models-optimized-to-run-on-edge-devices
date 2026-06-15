# diagnose_map_gap.py
#
# Diagnoses the gap between Ultralytics val mAP (~0.91) and our pycocotools
# benchmark mAP (~0.316) on the same model and same validation set.
# Ayni model ve ayni validation setinde Ultralytics val mAP (~0.91) ile bizim
# pycocotools benchmark mAP (~0.316) arasindaki farki teshis eder.
#
# Runs three checks / Uc kontrol calistirir:
#     1. Ultralytics native model.val() on restratified val.
#     1. Restratified val uzerinde Ultralytics native model.val().
#     2. Manual prediction sample: what does the model output look like?
#     2. Manuel tahmin ornegi: model ciktisi neye benziyor?
#     3. Class name + id consistency between YOLO model and COCO GT.
#     3. YOLO model ile COCO GT arasinda sinif adi + id tutarliligi.
#
# Usage / Kullanim:
#     python diagnose_map_gap.py

from pathlib import Path

from ultralytics import YOLO
from pycocotools.coco import COCO


PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

MODEL_PT = PROJECT_ROOT / "runs" / "yolov8l_restratified_test_14.06.2026" / "weights" / "best.pt"
YOLO_DATA = PROJECT_ROOT / "dataset" / "restratified_yolo" / "data.yaml"
COCO_VAL_GT = PROJECT_ROOT / "dataset" / "restratified_coco" / "val" / "_annotations.coco.json"


def check_1_ultralytics_val():
    # Run Ultralytics native validation on the restratified val split.
    # Restratified val split uzerinde Ultralytics native validation calistir.
    print("=" * 70)
    print("  CHECK 1: Ultralytics native model.val() on restratified val")
    print("=" * 70)

    model = YOLO(str(MODEL_PT))
    # split='val' uses the val: entry in data.yaml.
    # split='val', data.yaml'daki val: girdisini kullanir.
    metrics = model.val(data=str(YOLO_DATA), split="val", verbose=False)

    print(f"\n  Ultralytics mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  Ultralytics mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Per-class mAP@0.5        : {metrics.box.ap50}")
    print(f"  Class names              : {model.names}")
    return metrics


def check_2_class_consistency():
    # Compare YOLO model class names/order with COCO GT category names/ids.
    # YOLO model sinif adlari/sirasi ile COCO GT kategori adlari/id'lerini karsilastir.
    print("\n" + "=" * 70)
    print("  CHECK 2: Class name + id consistency")
    print("=" * 70)

    model = YOLO(str(MODEL_PT))
    print(f"\n  YOLO model.names: {model.names}")

    coco = COCO(str(COCO_VAL_GT))
    cats = coco.loadCats(coco.getCatIds())
    print(f"  COCO categories:")
    for c in sorted(cats, key=lambda x: x["id"]):
        print(f"    id={c['id']}  name={c['name']}")

    # Show how name-based mapping resolves.
    # Isim bazli eslemenin nasil cozuldugunu goster.
    coco_name_to_id = {c["name"].lower(): c["id"] for c in cats}
    print(f"\n  Name-based mapping (YOLO idx -> COCO id):")
    for yolo_idx, yolo_name in model.names.items():
        coco_id = coco_name_to_id.get(yolo_name.lower(), "NOT FOUND")
        print(f"    YOLO {yolo_idx} ({yolo_name}) -> COCO id {coco_id}")


def check_3_gt_annotation_sample():
    # Inspect a few GT annotations to verify bbox format and category ids.
    # Bbox formatini ve kategori id'lerini dogrulamak icin birkac GT annotation incele.
    print("\n" + "=" * 70)
    print("  CHECK 3: COCO GT annotation sample")
    print("=" * 70)

    coco = COCO(str(COCO_VAL_GT))
    ann_ids = coco.getAnnIds()[:5]
    anns = coco.loadAnns(ann_ids)
    print(f"\n  Total GT annotations: {len(coco.getAnnIds())}")
    print(f"  Sample annotations (first 5):")
    for a in anns:
        print(f"    image_id={a['image_id']}  category_id={a['category_id']}  "
              f"bbox={a['bbox']}  area={a['area']}")

    # Category id distribution in GT.
    # GT'de kategori id dagilimi.
    from collections import Counter
    cat_counts = Counter(a["category_id"] for a in coco.loadAnns(coco.getAnnIds()))
    print(f"\n  GT category_id distribution: {dict(cat_counts)}")


def main():
    print("\nDiagnosing mAP gap: Ultralytics (~0.91) vs pycocotools (~0.316)\n")

    check_2_class_consistency()
    check_3_gt_annotation_sample()
    check_1_ultralytics_val()

    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print("  If CHECK 1 shows ~0.91, Ultralytics agrees with training.")
    print("  Eger CHECK 1 ~0.91 gosterirse, Ultralytics egitimle uyumlu.")
    print("  Then the gap is in our pycocotools pipeline (CHECK 2/3 reveal why).")
    print("  O zaman fark bizim pycocotools pipeline'inda (CHECK 2/3 nedenini gosterir).")
    print("  Watch for: class mapping mismatch, bbox format, or category_id offset.")
    print("  Dikkat: sinif eslesme hatasi, bbox formati veya category_id kaymasi.")


if __name__ == "__main__":
    main()
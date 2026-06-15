# diagnose_predictions.py
#
# Checks how many YOLO predictions actually match GT images, and inspects
# the prediction/GT alignment that feeds pycocotools.
# YOLO tahminlerinin kacinin GT goruntuleriyle eslestigini kontrol eder ve
# pycocotools'a giden tahmin/GT hizalamasini inceler.
#
# Usage / Kullanim:
#     python diagnose_predictions.py

from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from pycocotools.coco import COCO


PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

MODEL_PT = PROJECT_ROOT / "runs" / "yolov8l_restratified_test_14.06.2026" / "weights" / "best.pt"
COCO_VAL_DIR = PROJECT_ROOT / "dataset" / "restratified_coco" / "val"
COCO_VAL_IMAGES = COCO_VAL_DIR / "images"
COCO_VAL_GT = COCO_VAL_DIR / "_annotations.coco.json"


def main():
    print("Loading GT...")
    coco_gt = COCO(str(COCO_VAL_GT))

    # GT filenames and image count.
    # GT dosya adlari ve goruntu sayisi.
    gt_fnames = {img["file_name"] for img in coco_gt.dataset["images"]}
    print(f"GT images: {len(gt_fnames)}")

    # Files physically in the COCO val images dir.
    # COCO val images dizinindeki fiziksel dosyalar.
    disk_fnames = {p.name for p in COCO_VAL_IMAGES.iterdir()}
    print(f"Disk images in COCO val: {len(disk_fnames)}")

    # Overlap between GT and disk.
    # GT ile disk arasi ortusme.
    print(f"GT ∩ disk: {len(gt_fnames & disk_fnames)}")
    print(f"In GT but not on disk: {len(gt_fnames - disk_fnames)}")
    print(f"On disk but not in GT: {len(disk_fnames - gt_fnames)}")

    # Run prediction on the COCO val images dir.
    # COCO val images dizininde tahmin calistir.
    print("\nRunning YOLO predict on COCO val images...")
    model = YOLO(str(MODEL_PT))
    results = model.predict(
        source=str(COCO_VAL_IMAGES),
        conf=0.001,
        iou=0.7,
        save=False,
        verbose=False,
        stream=True,
    )

    fname_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

    n_results = 0
    n_matched = 0
    n_unmatched = 0
    n_total_boxes = 0
    unmatched_examples = []

    for r in results:
        n_results += 1
        fname = Path(r.path).name
        if fname in fname_to_id:
            n_matched += 1
        else:
            n_unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append(fname)
        if r.boxes is not None:
            n_total_boxes += len(r.boxes)

    print(f"\n=== PREDICTION MATCHING ===")
    print(f"Total predicted images: {n_results}")
    print(f"  Matched to GT:   {n_matched}")
    print(f"  Unmatched (dropped): {n_unmatched}")
    print(f"Total predicted boxes: {n_total_boxes}")
    print(f"Avg boxes/image: {n_total_boxes / max(1, n_results):.1f}")

    if unmatched_examples:
        print(f"\nUnmatched filename examples:")
        for f in unmatched_examples:
            print(f"  {f}")
        print("GT filename examples:")
        for f in list(gt_fnames)[:5]:
            print(f"  {f}")

    # GT box count for comparison.
    # Karsilastirma icin GT box sayisi.
    print(f"\n=== GT vs PREDICTION VOLUME ===")
    print(f"GT total boxes:        {len(coco_gt.getAnnIds())}")
    print(f"Predicted total boxes: {n_total_boxes}")
    print(f"Ratio (pred/GT): {n_total_boxes / max(1, len(coco_gt.getAnnIds())):.2f}")
    print("\nIf predicted boxes >> GT boxes, conf=0.001 floods low-quality boxes (normal for mAP).")
    print("If matched << total, filename mismatch is dropping predictions (BUG).")


if __name__ == "__main__":
    main()
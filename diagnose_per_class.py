# diagnose_per_class.py
#
# Runs pycocotools COCOeval and reports per-class AP to find which class
# is collapsing. Strong suspect: category_id=0 (Person) being mishandled.
# pycocotools COCOeval calistirir ve hangi sinifin coKtugunu bulmak icin
# per-class AP raporlar. Guclu supheli: category_id=0 (Person) yanlis islenmesi.
#
# Usage / Kullanim:
#     python diagnose_per_class.py

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

MODEL_PT = PROJECT_ROOT / "runs" / "yolov8l_restratified_test_14.06.2026" / "weights" / "best.pt"
COCO_VAL_DIR = PROJECT_ROOT / "dataset" / "restratified_coco" / "val"
COCO_VAL_IMAGES = COCO_VAL_DIR / "images"
COCO_VAL_GT = COCO_VAL_DIR / "_annotations.coco.json"


def build_predictions(coco_gt):
    # Run YOLO and convert to COCO detection format with name-based mapping.
    # YOLO calistir ve isim bazli esleme ile COCO detection formatina cevir.
    model = YOLO(str(MODEL_PT))
    fname_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}

    coco_cats = coco_gt.loadCats(coco_gt.getCatIds())
    coco_name_to_id = {c["name"].lower(): c["id"] for c in coco_cats}

    results = model.predict(
        source=str(COCO_VAL_IMAGES), conf=0.001, iou=0.7,
        save=False, verbose=False, stream=True,
    )

    # Map other_vehicle -> othervehicle by removing underscore.
    # other_vehicle -> othervehicle alt cizgi kaldirarak esle.
    def norm(name):
        return name.lower().replace("_", "")

    coco_name_to_id_norm = {norm(c["name"]): c["id"] for c in coco_cats}

    preds = []
    yolo_to_coco = None
    for r in results:
        if yolo_to_coco is None:
            yolo_to_coco = {}
            for idx, nm in r.names.items():
                yolo_to_coco[idx] = coco_name_to_id_norm.get(norm(nm), idx)
            print(f"  Mapping used: {yolo_to_coco}")
        fname = Path(r.path).name
        img_id = fname_to_id.get(fname)
        if img_id is None or r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            preds.append({
                "image_id": int(img_id),
                "category_id": int(yolo_to_coco.get(int(cls[i]), int(cls[i]))),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
            })
    return preds


def main():
    coco_gt = COCO(str(COCO_VAL_GT))
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    print(f"COCO category ids: {cat_ids}")
    print(f"COCO categories: {[(c['id'], c['name']) for c in cats]}")

    # Distribution of category_id in predictions vs GT.
    # Tahminlerde vs GT'de category_id dagilimi.
    print("\nBuilding predictions...")
    preds = build_predictions(coco_gt)
    from collections import Counter
    pred_cats = Counter(p["category_id"] for p in preds)
    gt_cats = Counter(a["category_id"] for a in coco_gt.loadAnns(coco_gt.getAnnIds()))
    print(f"\n  GT category_id counts:   {dict(gt_cats)}")
    print(f"  Pred category_id counts: {dict(pred_cats)}")

    # Run COCOeval.
    # COCOeval calistir.
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Per-class AP @0.5.
    # Sinif basina AP @0.5.
    print("\n" + "=" * 70)
    print("  PER-CLASS AP@0.5 (pycocotools)")
    print("=" * 70)
    precision = coco_eval.eval["precision"]  # [TxRxKxAxM]
    # IoU=0.5 is index 0, area=all index 0, maxDets=100 index 2.
    # IoU=0.5 indeks 0, area=all indeks 0, maxDets=100 indeks 2.
    for k, c in enumerate(sorted(cats, key=lambda x: x["id"])):
        p = precision[0, :, k, 0, 2]
        p = p[p > -1]
        ap = float(p.mean()) if len(p) > 0 else float("nan")
        print(f"  {c['name']:<16} (id={c['id']}, k_index={k}): AP@0.5 = {ap:.4f}")

    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print("  Compare to Ultralytics per-class: car=0.84, person=0.95, other=0.97")
    print("  Ultralytics per-class ile karsilastir: car=0.84, person=0.95, other=0.97")
    print("  If one class shows ~0 here but high in Ultralytics -> that class id")
    print("  is mishandled by pycocotools (likely category_id=0 problem).")
    print("  Eger bir sinif burada ~0 ama Ultralytics'te yuksekse -> o sinif id'si")
    print("  pycocotools tarafindan yanlis isleniyor (muhtemelen category_id=0 sorunu).")


if __name__ == "__main__":
    main()
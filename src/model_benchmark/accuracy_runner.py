# accuracy_runner.py
#
# Test set accuracy evaluation using pycocotools standard.
# pycocotools standardi ile test set dogruluk degerlendirmesi.
#
# Both YOLO and Faster R-CNN predictions are converted to COCO format
# and evaluated with the same COCOeval pipeline for fair comparison.
# Hem YOLO hem Faster R-CNN tahminleri COCO formatina donusturulur ve
# adil karsilastirma icin ayni COCOeval pipeline'i ile degerlendirilir.

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRCNN_DIR = PROJECT_ROOT / "models" / "faster_rcnn"


# YOLO accuracy / YOLO dogruluk.

def _yolo_predictions_to_coco(results, coco_gt) -> list:
    # Convert Ultralytics results to COCO detection format.
    # Ultralytics sonuclarini COCO detection formatina cevirir.
    #
    # Maps YOLO class indices to COCO category ids using the GT annotation.
    # YOLO sinif indekslerini GT annotation kullanarak COCO category id'lerine esler.
    coco_results = []

    # Build filename -> image_id mapping from GT.
    # GT'den dosyaadi -> image_id eslesmesi olustur.
    fname_to_id = {}
    for img_info in coco_gt.dataset["images"]:
        fname_to_id[img_info["file_name"]] = img_info["id"]

    # Build YOLO index -> COCO category_id mapping.
    # YOLO indeks -> COCO category_id eslesmesi olustur.
    # Build YOLO class name -> COCO category_id mapping.
    # YOLO sinif adi -> COCO category_id eslesmesi olustur.
    coco_cats = coco_gt.loadCats(coco_gt.getCatIds())
    coco_name_to_id = {c["name"].lower(): c["id"] for c in coco_cats}

    # Get YOLO class names from first result.
    # Ilk sonuctan YOLO sinif isimlerini al.
    yolo_names = {}
    if results and hasattr(results[0], "names"):
        yolo_names = results[0].names

    # Map YOLO index -> COCO category_id via name matching.
    # Isim eslesmesi ile YOLO indeks -> COCO category_id esle.
    yolo_to_coco = {}
    for yolo_idx, yolo_name in yolo_names.items():
        yolo_to_coco[yolo_idx] = coco_name_to_id.get(yolo_name.lower(), yolo_idx)

    for r in results:
        fname = Path(r.path).name
        img_id = fname_to_id.get(fname)
        if img_id is None:
            continue

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy   = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls    = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            yolo_cls = int(cls[i])
            cat_id = yolo_to_coco.get(yolo_cls, yolo_cls)

            coco_results.append({
                "image_id":    int(img_id),
                "category_id": int(cat_id),
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(scores[i]),
            })

    return coco_results


def run_yolo_accuracy(model, test_images_dir: str, coco_gt: COCO) -> dict:
    # Run YOLO inference and evaluate with pycocotools.
    # YOLO inference calistir ve pycocotools ile degerlendir.
    results = model.predict(
        source=test_images_dir,
        conf=0.001,
        iou=0.7,
        save=False,
        verbose=False,
        stream=True,
    )

    # Collect all results from generator.
    # Generator'dan tum sonuclari topla.
    all_results = list(results)

    coco_preds = _yolo_predictions_to_coco(all_results, coco_gt)
    return _evaluate_coco(coco_gt, coco_preds)


# Faster R-CNN accuracy / Faster R-CNN dogruluk.

def _load_frcnn_test_loader(test_dir: str, batch_size: int = 8) -> DataLoader:
    # Build test DataLoader using our COCO dataset class.
    # Bizim COCO dataset sinifimizi kullanarak test DataLoader olustur.
    if str(FRCNN_DIR) not in sys.path:
        sys.path.insert(0, str(FRCNN_DIR))
    from dataset import CocoDetectionDataset, build_eval_transform, collate_fn

    ds = CocoDetectionDataset(
        images_dir=str(Path(test_dir) / "images"),
        annotation_path=str(Path(test_dir) / "_annotations.coco.json"),
        transform=build_eval_transform(),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    return loader


def _frcnn_predictions_to_coco(predictions: list, image_ids: list) -> list:
    # Convert Faster R-CNN predictions to COCO format.
    # Faster R-CNN tahminlerini COCO formatina cevirir.
    #
    # Labels are 1-based in Faster R-CNN (background=0), COCO expects 0-based.
    # Faster R-CNN'de etiketler 1-based (background=0), COCO 0-based bekler.
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        boxes  = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            coco_results.append({
                "image_id":    int(img_id),
                "category_id": int(label) - 1,
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(score),
            })
    return coco_results


def run_frcnn_accuracy(
    model, test_dir: str, coco_gt: COCO, device: str = "cuda:0",
) -> dict:
    # Run Faster R-CNN inference and evaluate with pycocotools.
    # Faster R-CNN inference calistir ve pycocotools ile degerlendir.
    loader = _load_frcnn_test_loader(test_dir)

    all_coco_results = []
    model.eval()

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        with torch.no_grad():
            preds = model(images_dev)
        image_ids = [int(t["image_id"].item()) for t in targets]
        all_coco_results.extend(_frcnn_predictions_to_coco(preds, image_ids))

    return _evaluate_coco(coco_gt, all_coco_results)


# Shared COCO evaluation / Ortak COCO degerlendirme.

def _evaluate_coco(coco_gt: COCO, coco_results: list) -> dict:
    # Run COCOeval and extract per-class metrics.
    # COCOeval calistir ve per-class metrikleri cikar.
    if not coco_results:
        return _empty_result(coco_gt)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50    = float(coco_eval.stats[1])
    map50_95 = float(coco_eval.stats[0])

    # Per-class extraction.
    # Per-class cikarma.
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    precision = coco_eval.eval["precision"]

    per_class = {}
    for k, name in enumerate(class_names):
        p50 = precision[0, :, k, 0, 2]
        p50 = p50[p50 > -1]
        ap50 = float(p50.mean()) if len(p50) > 0 else 0.0

        p_all = precision[:, :, k, 0, 2]
        p_all = p_all[p_all > -1]
        ap_all = float(p_all.mean()) if len(p_all) > 0 else 0.0

        per_class[name] = {"map50": ap50, "map50_95": ap_all}

    return {
        "map50":     map50,
        "map50_95":  map50_95,
        "per_class": per_class,
    }


def _empty_result(coco_gt: COCO) -> dict:
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_names = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    per_class = {n: {"map50": 0.0, "map50_95": 0.0} for n in class_names}
    return {"map50": 0.0, "map50_95": 0.0, "per_class": per_class}
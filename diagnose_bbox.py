# diagnose_bbox.py
#
# Checks bbox coordinate scale alignment between YOLO predictions and COCO GT.
# YOLO tahminleri ile COCO GT arasinda bbox koordinat olcek hizalamasini kontrol eder.
#
# Verifies / Dogrular:
#     - Real image dimensions vs GT-recorded width/height.
#     - Gercek goruntu boyutlari vs GT'de kayitli width/height.
#     - YOLO predicted xyxy range vs GT bbox range for the same image.
#     - Ayni goruntu icin YOLO xyxy araligi vs GT bbox araligi.
#
# Usage / Kullanim:
#     python diagnose_bbox.py

from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from pycocotools.coco import COCO


PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

MODEL_PT = PROJECT_ROOT / "runs" / "yolov8l_restratified_test_14.06.2026" / "weights" / "best.pt"
COCO_VAL_DIR = PROJECT_ROOT / "dataset" / "restratified_coco" / "val"
COCO_VAL_IMAGES = COCO_VAL_DIR / "images"
COCO_VAL_GT = COCO_VAL_DIR / "_annotations.coco.json"


def main():
    coco_gt = COCO(str(COCO_VAL_GT))

    # Check real image sizes vs GT-recorded sizes for a few sources.
    # Birkac kaynak icin gercek goruntu boyutlari vs GT-kayitli boyutlar.
    print("=" * 70)
    print("  IMAGE SIZE: real (on disk) vs GT-recorded")
    print("=" * 70)

    imgs = coco_gt.dataset["images"]
    # Pick samples from different sources.
    # Farkli kaynaklardan ornek sec.
    seen_sources = {}
    for img in imgs:
        fname = img["file_name"]
        src = fname.split("_")[0]
        if "rgbt" in fname:
            src = "rgbt_tiny"
        elif "thermal_v1_alt" in fname:
            src = "thermal_v1_alt"
        elif "thermal_v1" in fname:
            src = "thermal_v1"
        elif "hituav" in fname:
            src = "hituav"
        elif "drone" in fname:
            src = "drone_thermal"
        if src in seen_sources:
            continue
        seen_sources[src] = img
        if len(seen_sources) >= 5:
            break

    for src, img in seen_sources.items():
        disk_path = COCO_VAL_IMAGES / img["file_name"]
        if disk_path.exists():
            real_w, real_h = Image.open(disk_path).size
        else:
            real_w, real_h = "?", "?"
        gt_w, gt_h = img["width"], img["height"]
        match = "OK" if (real_w == gt_w and real_h == gt_h) else "MISMATCH"
        print(f"  {src:<16} {img['file_name']:<28} "
              f"real={real_w}x{real_h}  GT={gt_w}x{gt_h}  [{match}]")

    # For one image, compare predicted xyxy vs GT bbox ranges.
    # Bir goruntu icin tahmin xyxy vs GT bbox araliklarini karsilastir.
    print("\n" + "=" * 70)
    print("  BBOX RANGE: prediction vs GT (same image)")
    print("=" * 70)

    # Use an rgbt_tiny image (dominant source).
    # rgbt_tiny goruntusu kullan (baskin kaynak).
    target_img = seen_sources.get("rgbt_tiny") or list(seen_sources.values())[0]
    target_fname = target_img["file_name"]
    target_id = target_img["id"]
    print(f"\n  Target image: {target_fname} (id={target_id})")
    print(f"  GT size: {target_img['width']}x{target_img['height']}")

    # GT boxes for this image.
    # Bu goruntu icin GT kutular.
    ann_ids = coco_gt.getAnnIds(imgIds=[target_id])
    anns = coco_gt.loadAnns(ann_ids)
    print(f"\n  GT boxes: {len(anns)}")
    if anns:
        xs = [a["bbox"][0] for a in anns]
        ys = [a["bbox"][1] for a in anns]
        ws = [a["bbox"][2] for a in anns]
        hs = [a["bbox"][3] for a in anns]
        print(f"    x range: {min(xs):.1f} - {max(xs):.1f}")
        print(f"    y range: {min(ys):.1f} - {max(ys):.1f}")
        print(f"    w range: {min(ws):.1f} - {max(ws):.1f}")
        print(f"    h range: {min(hs):.1f} - {max(hs):.1f}")
        print(f"    Sample GT bbox [x,y,w,h]: {anns[0]['bbox']}")

    # Predict on this single image.
    # Bu tek goruntude tahmin yap.
    model = YOLO(str(MODEL_PT))
    results = model.predict(
        source=str(COCO_VAL_IMAGES / target_fname),
        conf=0.25,  # higher conf for clean comparison / temiz karsilastirma icin yuksek conf
        iou=0.7,
        save=False,
        verbose=False,
    )
    r = results[0]
    print(f"\n  Predicted boxes (conf>0.25): {len(r.boxes)}")
    print(f"  Result orig_shape: {r.orig_shape}")
    if len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        print(f"    x1 range: {xyxy[:,0].min():.1f} - {xyxy[:,0].max():.1f}")
        print(f"    y1 range: {xyxy[:,1].min():.1f} - {xyxy[:,1].max():.1f}")
        print(f"    x2 range: {xyxy[:,2].min():.1f} - {xyxy[:,2].max():.1f}")
        print(f"    y2 range: {xyxy[:,3].min():.1f} - {xyxy[:,3].max():.1f}")
        # Convert first to xywh for direct comparison.
        # Dogrudan karsilastirma icin ilkini xywh'ye cevir.
        x1, y1, x2, y2 = xyxy[0]
        print(f"    Sample pred bbox [x,y,w,h]: [{x1:.1f}, {y1:.1f}, {x2-x1:.1f}, {y2-y1:.1f}]")

    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print("  If real size != GT size -> coordinates are on different scales (BUG).")
    print("  Eger gercek boyut != GT boyut -> koordinatlar farkli olcekte (BUG).")
    print("  If pred xyxy max >> GT size -> predictions in wrong scale.")
    print("  Eger pred xyxy max >> GT boyut -> tahminler yanlis olcekte.")
    print("  GT and pred bbox ranges should overlap if scales match.")
    print("  Olcekler uyusursa GT ve pred bbox araliklari ortusmeli.")


if __name__ == "__main__":
    main()
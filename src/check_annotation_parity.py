"""
COCO ↔ YOLO annotation sayısı parity kontrol.

Her split için:
  - COCO JSON'daki annotation sayısı
  - YOLO labels/{split}/*.txt satır sayısı (= annotation sayısı)
  - Sınıf bazlı dağılım
karşılaştırılır. Eşit değilse mismatch raporlanır.
"""

import json
import os
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────
# KONFİGÜRASYON
# ──────────────────────────────────────────────

DATASET_ROOT = "/home/ugo/Documents/Python/uc_cihaz_obejct_detection/CNN-models-optimized-to-run-on-edge-devices/dataset"

COCO_DIR = f"{DATASET_ROOT}/dataset_augmented_04_23_2026/dataset_augmented"
YOLO_DIR = f"{DATASET_ROOT}/dataset_augmented_yolo_23.04.2026/dataset_augmented_yolo"

SPLITS = ["train", "val", "test"]


def count_coco(split):
    """COCO: annotation sayısı + görüntü sayısı + sınıf bazlı sayım."""
    json_path = os.path.join(COCO_DIR, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    ann_total = len(data["annotations"])
    img_total = len(data["images"])
    img_with_ann = len({ann["image_id"] for ann in data["annotations"]})
    per_class = Counter(ann["category_id"] for ann in data["annotations"])
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    return {
        "ann": ann_total,
        "img": img_total,
        "img_with_ann": img_with_ann,
        "per_class": per_class,
        "cat_names": cat_names,
    }


def count_yolo(split):
    """YOLO: label satır sayısı + label dosya sayısı + image dosya sayısı."""
    label_dir = Path(YOLO_DIR) / "labels" / split
    image_dir = Path(YOLO_DIR) / "images" / split
    if not label_dir.exists():
        return None

    ann_total = 0
    per_class = Counter()
    label_files = list(label_dir.glob("*.txt"))

    for txt in label_files:
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                per_class[cls] += 1
                ann_total += 1

    img_files = 0
    if image_dir.exists():
        img_files = sum(
            1 for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

    return {
        "ann": ann_total,
        "img": img_files,
        "label_files": len(label_files),
        "per_class": per_class,
    }


def main():
    print("=" * 70)
    print("COCO ↔ YOLO ANNOTATION PARITY KONTROLÜ")
    print("=" * 70)
    print(f"COCO: {COCO_DIR}")
    print(f"YOLO: {YOLO_DIR}")
    print("=" * 70)

    grand_coco_ann = 0
    grand_yolo_ann = 0
    grand_coco_img = 0
    grand_yolo_img = 0
    any_mismatch = False

    for split in SPLITS:
        print(f"\n[{split.upper()}]")

        coco = count_coco(split)
        yolo = count_yolo(split)

        if coco is None:
            print(f"  [WARN] COCO JSON yok: {split}")
            continue
        if yolo is None:
            print(f"  [WARN] YOLO labels/{split} yok")
            continue

        # Görüntü kıyası
        print(f"  COCO görüntü     : {coco['img']}  ({coco['img_with_ann']} tanesi annotated)")
        print(f"  YOLO görüntü     : {yolo['img']}  ({yolo['label_files']} txt dosya)")
        img_diff = yolo["img"] - coco["img"]
        lbl_diff = yolo["label_files"] - coco["img_with_ann"]
        print(f"  Görüntü fark     : {img_diff:+d}   |   txt vs annotated img fark: {lbl_diff:+d}")

        # Annotation kıyası
        print(f"  COCO annotation  : {coco['ann']}")
        print(f"  YOLO annotation  : {yolo['ann']}")
        ann_diff = yolo["ann"] - coco["ann"]
        status = "EŞİT ✓" if ann_diff == 0 else f"FARK = {ann_diff:+d}  ✗"
        print(f"  Annotation durum : {status}")

        if ann_diff != 0 or img_diff != 0 or lbl_diff != 0:
            any_mismatch = True

        # Sınıf bazlı kıyas
        coco_pc, yolo_pc = coco["per_class"], yolo["per_class"]
        cat_names = coco["cat_names"]
        all_ids = sorted(set(coco_pc) | set(yolo_pc))
        print(f"  {'cls_id':>6} | {'name':<16} | {'coco':>8} | {'yolo':>8} | {'fark':>6}")
        print(f"  {'-'*6} | {'-'*16} | {'-'*8} | {'-'*8} | {'-'*6}")
        for cid in all_ids:
            name = cat_names.get(cid, "?")
            c = coco_pc.get(cid, 0)
            y = yolo_pc.get(cid, 0)
            d = y - c
            mark = "" if d == 0 else "  ✗"
            print(f"  {cid:>6} | {name:<16} | {c:>8} | {y:>8} | {d:>+6}{mark}")

        grand_coco_ann += coco["ann"]
        grand_yolo_ann += yolo["ann"]
        grand_coco_img += coco["img"]
        grand_yolo_img += yolo["img"]

    print("\n" + "=" * 70)
    print(f"TOPLAM görüntü   COCO: {grand_coco_img}   YOLO: {grand_yolo_img}   "
          f"FARK: {grand_yolo_img - grand_coco_img:+d}")
    print(f"TOPLAM annotation COCO: {grand_coco_ann}   YOLO: {grand_yolo_ann}   "
          f"FARK: {grand_yolo_ann - grand_coco_ann:+d}")
    print("=" * 70)

    if any_mismatch:
        print("\n[SONUÇ] Mismatch var. Olası sebepler:")
        print("  1. coco_to_yolo_transform.py append mode → tekrar run dup")
        print("  2. category_id mapping kayması (0-index vs 1-index)")
        print("  3. Eksik image_id → ann atlanmış olabilir")
    else:
        print("\n[SONUÇ] Tüm splitler eşit ✓")


if __name__ == "__main__":
    main()

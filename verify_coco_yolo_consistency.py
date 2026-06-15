# verify_coco_yolo_consistency.py
#
# Verifies COCO category_id vs YOLO class_id consistency across the WHOLE
# dataset by comparing per-image class sets. Confirms whether YOLO label
# indices match COCO category ids for every image and every source.
# TUM dataset boyunca COCO category_id vs YOLO class_id tutarliligini, goruntu
# basina sinif kumelerini karsilastirarak dogrular. YOLO label indekslerinin
# her goruntu ve her kaynak icin COCO category id'leriyle eslesip eslesmedigini
# kesinlestirir.
#
# Method / Yontem:
#     For each image, compare the SET of class ids in COCO GT vs YOLO label.
#     Her goruntu icin COCO GT'deki sinif id KUMESINI YOLO label ile karsilastir.
#     If they match for all images, indices are consistent.
#     Tum goruntulerde eslesirse, indeksler tutarli.
#
# Usage / Kullanim:
#     python verify_coco_yolo_consistency.py

from pathlib import Path
from collections import defaultdict, Counter

from pycocotools.coco import COCO


PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

# Use the restratified val split (has all sources).
# Restratified val split kullan (tum kaynaklar var).
COCO_DIR = PROJECT_ROOT / "dataset" / "restratified_coco" / "val"
YOLO_LABELS = PROJECT_ROOT / "dataset" / "restratified_yolo" / "labels" / "val"


def source_of(fname):
    if "rgbt_tiny" in fname:
        return "rgbt_tiny"
    if "thermal_v1_alt" in fname:
        return "thermal_v1_alt"
    if "thermal_v1" in fname:
        return "thermal_v1"
    if "hituav" in fname:
        return "hituav"
    if "drone_thermal" in fname:
        return "drone_thermal"
    return "unknown"


def main():
    coco = COCO(str(COCO_DIR / "_annotations.coco.json"))

    # Build per-image COCO class-id set.
    # Goruntu basina COCO sinif-id kumesi olustur.
    coco_classes_by_img = defaultdict(set)
    img_id_to_name = {}
    for img in coco.dataset["images"]:
        img_id_to_name[img["id"]] = img["file_name"]
    for ann in coco.dataset["annotations"]:
        fname = img_id_to_name[ann["image_id"]]
        coco_classes_by_img[fname].add(ann["category_id"])

    # Compare each image's COCO set vs YOLO set.
    # Her goruntunun COCO kumesi vs YOLO kumesini karsilastir.
    match = 0
    mismatch = 0
    mismatch_examples = []
    by_source_match = defaultdict(lambda: [0, 0])  # [match, mismatch]

    # Also track cross-tabulation: when COCO has class X, what YOLO class appears?
    # Capraz tablo: COCO sinif X varken YOLO hangi sinifi gosteriyor?
    # Only meaningful for single-class images.
    # Sadece tek-sinifli goruntuler icin anlamli.
    single_class_xtab = defaultdict(Counter)

    for fname, coco_set in coco_classes_by_img.items():
        yolo_label = YOLO_LABELS / (Path(fname).stem + ".txt")
        if not yolo_label.exists():
            continue
        yolo_set = set()
        for line in yolo_label.read_text().strip().split("\n"):
            if line.strip():
                yolo_set.add(int(line.split()[0]))

        src = source_of(fname)
        if coco_set == yolo_set:
            match += 1
            by_source_match[src][0] += 1
        else:
            mismatch += 1
            by_source_match[src][1] += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append((fname, sorted(coco_set), sorted(yolo_set)))

        # Single-class cross-tab.
        # Tek-sinif capraz tablo.
        if len(coco_set) == 1 and len(yolo_set) == 1:
            coco_c = list(coco_set)[0]
            yolo_c = list(yolo_set)[0]
            single_class_xtab[coco_c][yolo_c] += 1

    print("=" * 70)
    print("  COCO vs YOLO CLASS-SET CONSISTENCY (per image)")
    print("=" * 70)
    print(f"\n  Matching images:    {match}")
    print(f"  Mismatching images: {mismatch}")
    print(f"  Match rate: {match / max(1, match + mismatch) * 100:.1f}%")

    print(f"\n  Per-source match/mismatch:")
    for src, (m, mm) in sorted(by_source_match.items()):
        print(f"    {src:<16} match={m:>5}  mismatch={mm:>5}")

    if mismatch_examples:
        print(f"\n  Mismatch examples:")
        for fname, c, y in mismatch_examples:
            print(f"    {fname}: COCO={c}  YOLO={y}")

    print("\n" + "=" * 70)
    print("  SINGLE-CLASS CROSS-TAB (COCO class -> YOLO class)")
    print("=" * 70)
    print("  For images with exactly one class in both formats.")
    print("  Her iki formatta tam bir sinif olan goruntuler icin.")
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    yolo_names = {0: "car?", 1: "person?", 2: "other?"}
    for coco_c in sorted(single_class_xtab.keys()):
        print(f"\n  COCO {coco_c} ({cats.get(coco_c)}):")
        for yolo_c, cnt in sorted(single_class_xtab[coco_c].items()):
            print(f"    -> YOLO index {yolo_c}: {cnt} images")

    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    print("  If match rate ~100% -> COCO and YOLO use SAME index order.")
    print("  Eger match orani ~100% -> COCO ve YOLO AYNI index sirasini kullanir.")
    print("  Cross-tab shows the true meaning of each YOLO index.")
    print("  Capraz tablo her YOLO index'inin gercek anlamini gosterir.")
    print("  E.g. if COCO 0 (Person) -> YOLO 0 always, then YOLO 0 = Person.")
    print("  Orn. COCO 0 (Person) -> hep YOLO 0 ise, YOLO 0 = Person demektir.")


if __name__ == "__main__":
    main()
# analyze_source_difficulty.py
#
# Analyzes test set difficulty broken down by data source prefix.
# Test seti zorlugunu veri kaynagi prefixine gore ayirarak analiz eder.
#
# Part 1 (no model needed): per-source image count, object count,
# objects-per-image, and class distribution.
# Bolum 1 (model gerekmez): kaynak basina goruntu sayisi, nesne sayisi,
# nesne-basina-goruntu ve sinif dagilimi.
#
# Part 2 (optional, needs a model): per-source mAP using a YOLO model.
# Bolum 2 (opsiyonel, model gerekir): YOLO modeli ile kaynak basina mAP.
#
# Usage / Kullanim:
#     python analyze_source_difficulty.py

import json
import re
from collections import defaultdict
from pathlib import Path


# Paths / Yollar
PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")
TEST_DIR = PROJECT_ROOT / "dataset" / "2x_augmented_coco_dataset" / "dataset_augmented" / "test"
TEST_ANN = TEST_DIR / "_annotations.coco.json"


# Known source prefixes / Bilinen kaynak prefixleri
SOURCES = ["rgbt_tiny", "thermal_v1_alt", "thermal_v1", "hituav", "drone_thermal"]


def source_of(filename: str) -> str:
    # Map a filename to its source prefix.
    # Bir dosya adini kaynak prefixine esler.
    #
    # Strip leading aug_ marker if present.
    # Varsa bastaki aug_ isaretini cikar.
    name = filename
    if name.startswith("aug_"):
        # aug_affine_0025335_drone_thermal_000001.jpg -> drone_thermal_000001.jpg
        # Find the source prefix after the aug id.
        # Aug id'sinden sonraki kaynak prefixini bul.
        for src in SOURCES:
            if src in name:
                return src
        return "unknown"
    # thermal_v1_alt must be checked before thermal_v1 (substring).
    # thermal_v1_alt, thermal_v1'den once kontrol edilmeli (alt dizi).
    for src in SOURCES:
        if name.startswith(src):
            return src
    return "unknown"


def main():
    print("Loading test annotations...")
    with open(TEST_ANN) as f:
        coco = json.load(f)

    # Build category id -> name map.
    # Kategori id -> isim haritasi olustur.
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Build image id -> filename map.
    # Goruntu id -> dosya adi haritasi olustur.
    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    # Per-source aggregation.
    # Kaynak basina toplama.
    src_images = defaultdict(set)
    src_obj_count = defaultdict(int)
    src_class_count = defaultdict(lambda: defaultdict(int))

    # Count images per source.
    # Kaynak basina goruntu say.
    for img in coco["images"]:
        src = source_of(img["file_name"])
        src_images[src].add(img["id"])

    # Count annotations per source.
    # Kaynak basina annotation say.
    for ann in coco["annotations"]:
        img_name = img_id_to_name.get(ann["image_id"], "")
        src = source_of(img_name)
        src_obj_count[src] += 1
        cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
        src_class_count[src][cat_name] += 1

    # Report.
    # Rapor.
    print("\n" + "=" * 70)
    print("  PER-SOURCE TEST SET ANALYSIS")
    print("=" * 70)

    all_classes = sorted(cat_id_to_name.values())

    header = f"\n{'Source':<18}{'Images':>8}{'Objects':>10}{'Obj/Img':>10}"
    for cn in all_classes:
        header += f"{cn:>14}"
    print(header)
    print("-" * len(header))

    for src in SOURCES + ["unknown"]:
        n_img = len(src_images[src])
        if n_img == 0:
            continue
        n_obj = src_obj_count[src]
        density = n_obj / n_img if n_img else 0
        row = f"{src:<18}{n_img:>8}{n_obj:>10}{density:>10.1f}"
        for cn in all_classes:
            c = src_class_count[src][cn]
            pct = (c / n_obj * 100) if n_obj else 0
            row += f"{c:>8} ({pct:>3.0f}%)"
        print(row)

    print("\n" + "=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print("If rgbt_tiny has much higher Obj/Img than other sources,")
    print("eger rgbt_tiny digerlerinden cok daha yuksek Obj/Img'ye sahipse,")
    print("that confirms it contains denser, harder scenes.")
    print("bu, daha yogun ve zor sahneler icerdigini dogrular.")


if __name__ == "__main__":
    main()
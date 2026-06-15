# restratify_dataset.py
#
# Re-splits the merged multi-source thermal dataset with source-stratified
# train/val/test partitioning, fixing the broken split where rgbt_tiny was
# absent from validation.
# Birlestirilmis cok-kaynakli thermal dataset'i kaynak-stratified train/val/test
# bolmesiyle yeniden boler. rgbt_tiny'nin validation'da bulunmadigi bozuk
# bolmeyi duzeltir.
#
# Strategy / Strateji:
#     - Collect all ORIGINAL frames (non-aug) from existing 3 splits.
#     - Tum ORIJINAL frame'leri (aug olmayan) mevcut 3 split'ten topla.
#     - Group by source prefix, split each source 70/15/15 (seed=42).
#     - Kaynak prefixine gore grupla, her kaynagi 70/15/15 bol (seed=42).
#     - Augmented copies follow their original; only train keeps augments.
#     - Augment kopyalar orijinalini takip eder; sadece train augment tutar.
#     - Rebuilds COCO annotations and copies YOLO labels in sync.
#     - COCO annotation'lari yeniden kurar ve YOLO label'lari senkron kopyalar.
#
# Output / Cikti:
#     dataset/restratified_coco/{train,val,test}/{images, _annotations.coco.json}
#     dataset/restratified_yolo/{images,labels}/{train,val,test} + data.yaml
#
# Usage / Kullanim:
#     python restratify_dataset.py

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# Configuration / Konfigurasyon
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = Path("/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection")

# Source datasets / Kaynak datasetler
COCO_SRC = PROJECT_ROOT / "dataset" / "2x_augmented_coco_dataset" / "dataset_augmented"
YOLO_SRC = PROJECT_ROOT / "dataset" / "2x_augmented_yolo_dataset" / "dataset_augmented_yolo"

# Output datasets / Cikti datasetler
COCO_OUT = PROJECT_ROOT / "dataset" / "restratified_coco"
YOLO_OUT = PROJECT_ROOT / "dataset" / "restratified_yolo"

SPLITS = ["train", "val", "test"]

# Known source prefixes, longest first to avoid substring collisions.
# Bilinen kaynak prefixleri, alt dizi cakismasini onlemek icin en uzun once.
SOURCES = ["thermal_v1_alt", "thermal_v1", "rgbt_tiny", "hituav", "drone_thermal"]

# Augment type markers / Augment tip isaretleri
AUG_TYPES = ["affine", "crop", "hflip", "rotate", "vflip"]


def is_augmented(filename: str) -> bool:
    # True if the file is an augmented copy.
    # Dosya augment kopyaysa True.
    return filename.startswith("aug_")


def original_of(filename: str) -> str:
    # Map any filename (aug or not) to its original frame filename.
    # Herhangi bir dosya adini (aug olsun olmasin) orijinal frame adina esler.
    #
    # aug_affine_0025335_drone_thermal_000001.jpg -> drone_thermal_000001.jpg
    # drone_thermal_000001.jpg                    -> drone_thermal_000001.jpg
    if not is_augmented(filename):
        return filename
    # Strip aug_<type>_<augid>_ prefix.
    # aug_<tip>_<augid>_ prefixini cikar.
    for src in SOURCES:
        idx = filename.find(src)
        if idx != -1:
            return filename[idx:]
    return filename


def source_of(filename: str) -> str:
    # Map a filename to its source prefix.
    # Bir dosya adini kaynak prefixine esler.
    orig = original_of(filename)
    for src in SOURCES:
        if orig.startswith(src):
            return src
    return "unknown"


def collect_all_files():
    # Walk all 3 source splits, return lists of original and augmented files.
    # 3 kaynak split'i tara, orijinal ve augment dosya listelerini dondur.
    #
    # Returns / Dondurur:
    #     originals: set of original filenames
    #     aug_by_original: dict original_filename -> list of aug filenames
    originals = set()
    aug_by_original = defaultdict(list)

    for split in SPLITS:
        img_dir = COCO_SRC / split / "images"
        for img_path in img_dir.iterdir():
            fname = img_path.name
            if is_augmented(fname):
                orig = original_of(fname)
                aug_by_original[orig].append(fname)
            else:
                originals.add(fname)

    return originals, aug_by_original


def split_originals(originals):
    # Source-stratified split of original frames into train/val/test.
    # Orijinal frame'lerin kaynak-stratified train/val/test bolmesi.
    #
    # Returns / Dondurur: dict split_name -> set of original filenames
    rng = random.Random(SEED)

    by_source = defaultdict(list)
    for fname in originals:
        by_source[source_of(fname)].append(fname)

    assignment = {"train": set(), "val": set(), "test": set()}

    print("\n[Split] Per-source partitioning:")
    for src in sorted(by_source.keys()):
        files = sorted(by_source[src])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Remainder goes to test for exactness.
        # Kalan, tamlik icin test'e gider.
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        assignment["train"].update(train_files)
        assignment["val"].update(val_files)
        assignment["test"].update(test_files)

        print(f"  {src:<16} total={n:>6}  "
              f"train={len(train_files):>6}  "
              f"val={len(val_files):>5}  "
              f"test={len(test_files):>5}")

    return assignment


def build_file_lists(assignment, aug_by_original):
    # For each split, build the full list of image filenames.
    # Her split icin tam goruntu dosya adi listesi olustur.
    #
    # Train gets originals + their augments. Val/test get only originals.
    # Train orijinalleri + augment'lerini alir. Val/test sadece orijinalleri.
    split_files = {"train": [], "val": [], "test": []}

    # Train: originals + augments.
    # Train: orijinaller + augment'ler.
    for orig in assignment["train"]:
        split_files["train"].append(orig)
        split_files["train"].extend(aug_by_original.get(orig, []))

    # Val and test: originals only.
    # Val ve test: sadece orijinaller.
    for split in ["val", "test"]:
        for orig in assignment[split]:
            split_files[split].append(orig)

    return split_files


def find_source_image(fname):
    # Locate a given image file across the 3 original COCO splits.
    # Verilen goruntu dosyasini 3 orijinal COCO split'inde bulur.
    for split in SPLITS:
        p = COCO_SRC / split / "images" / fname
        if p.exists():
            return p
    return None


def find_source_label(fname):
    # Locate the YOLO label .txt for a given image across the 3 splits.
    # Verilen goruntu icin YOLO label .txt'sini 3 split'te bulur.
    label_name = Path(fname).stem + ".txt"
    for split in SPLITS:
        p = YOLO_SRC / "labels" / split / label_name
        if p.exists():
            return p
    return None


def load_all_coco_annotations():
    # Load and merge COCO annotations from all 3 source splits.
    # 3 kaynak split'ten COCO annotation'lari yukle ve birlestir.
    #
    # Returns / Dondurur:
    #     categories: list (from train, all share the same)
    #     img_by_name: dict file_name -> {width, height}
    #     anns_by_name: dict file_name -> list of annotation dicts (no id/image_id)
    categories = None
    img_by_name = {}
    anns_by_name = defaultdict(list)

    for split in SPLITS:
        ann_path = COCO_SRC / split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        if categories is None:
            categories = coco["categories"]

        # image id -> file_name for this split.
        # Bu split icin image id -> file_name.
        id_to_name = {}
        for img in coco["images"]:
            id_to_name[img["id"]] = img["file_name"]
            img_by_name[img["file_name"]] = {
                "width": img["width"],
                "height": img["height"],
            }

        for ann in coco["annotations"]:
            fname = id_to_name.get(ann["image_id"])
            if fname is None:
                continue
            # Store annotation without id/image_id (reassigned later).
            # Annotation'i id/image_id olmadan sakla (sonra yeniden atanir).
            anns_by_name[fname].append({
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0),
            })

    return categories, img_by_name, anns_by_name


def write_coco_split(split, file_list, categories, img_by_name, anns_by_name):
    # Build and write a COCO annotation file for one split, copying images.
    # Bir split icin COCO annotation dosyasi olustur ve yaz, goruntuleri kopyala.
    out_img_dir = COCO_OUT / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for fname in sorted(file_list):
        src_img = find_source_image(fname)
        if src_img is None:
            print(f"  [WARN] image not found: {fname}")
            continue

        # Copy image.
        # Goruntuyu kopyala.
        shutil.copy2(src_img, out_img_dir / fname)

        meta = img_by_name.get(fname, {"width": 640, "height": 640})
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": meta["width"],
            "height": meta["height"],
        })

        for ann in anns_by_name.get(fname, []):
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "segmentation": ann["segmentation"],
                "iscrowd": ann["iscrowd"],
            })
            ann_id += 1

        img_id += 1

    coco_out = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(COCO_OUT / split / "_annotations.coco.json", "w") as f:
        json.dump(coco_out, f)

    print(f"  [COCO] {split}: {len(images)} images, {len(annotations)} annotations")
    return len(images), len(annotations)


def write_yolo_split(split, file_list):
    # Copy images and labels for one split into the YOLO output layout.
    # Bir split icin goruntu ve label'lari YOLO cikti duzenine kopyalar.
    out_img_dir = YOLO_OUT / "images" / split
    out_lbl_dir = YOLO_OUT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img, n_lbl, n_missing_lbl = 0, 0, 0

    for fname in sorted(file_list):
        # Image: find in YOLO source splits.
        # Goruntu: YOLO kaynak split'lerinde bul.
        src_img = None
        for s in SPLITS:
            p = YOLO_SRC / "images" / s / fname
            if p.exists():
                src_img = p
                break
        if src_img is None:
            continue
        shutil.copy2(src_img, out_img_dir / fname)
        n_img += 1

        # Label.
        # Label.
        src_lbl = find_source_label(fname)
        if src_lbl is not None:
            shutil.copy2(src_lbl, out_lbl_dir / (Path(fname).stem + ".txt"))
            n_lbl += 1
        else:
            # No label = background image (valid in YOLO).
            # Label yok = arka plan goruntusu (YOLO'da gecerli).
            n_missing_lbl += 1

    print(f"  [YOLO] {split}: {n_img} images, {n_lbl} labels, "
          f"{n_missing_lbl} without label")
    return n_img, n_lbl


def write_yolo_yaml():
    # Write data.yaml for the restratified YOLO dataset (same class order).
    # Yeniden bolunmus YOLO dataset'i icin data.yaml yaz (ayni sinif sirasi).
    yaml_content = f"""path: {YOLO_OUT}
train: images/train
val:   images/val
test:  images/test
nc: 3
names:
  0: car
  1: person
  2: other_vehicle
"""
    with open(YOLO_OUT / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"  [YOLO] data.yaml written")


def verify_no_leakage(assignment, aug_by_original):
    # Ensure no original frame appears in more than one split.
    # Hicbir orijinal frame'in birden fazla split'te olmadigini dogrula.
    print("\n[Verify] Leakage check (original frames)...")
    t, v, te = assignment["train"], assignment["val"], assignment["test"]
    tv = t & v
    tt = t & te
    vt = v & te
    print(f"  train ∩ val:  {len(tv)}")
    print(f"  train ∩ test: {len(tt)}")
    print(f"  val ∩ test:   {len(vt)}")
    if len(tv) == len(tt) == len(vt) == 0:
        print("  PASS: no original-frame leakage")
    else:
        print("  FAIL: leakage detected!")

    # Also verify augments only in train.
    # Ayrica augment'lerin sadece train'de oldugunu dogrula.
    print("\n[Verify] Augment placement (val/test must have zero augments)...")
    for split in ["val", "test"]:
        img_dir = COCO_OUT / split / "images"
        if img_dir.exists():
            aug_count = sum(1 for p in img_dir.iterdir() if is_augmented(p.name))
            status = "PASS" if aug_count == 0 else "FAIL"
            print(f"  {split}: {aug_count} augments  [{status}]")


def report_source_distribution():
    # Report per-source image counts in each new split.
    # Her yeni split'te kaynak basina goruntu sayilarini raporla.
    print("\n[Report] Source distribution in new splits (COCO images):")
    header = f"  {'Source':<16}" + "".join(f"{s:>10}" for s in SPLITS)
    print(header)
    for src in SOURCES:
        row = f"  {src:<16}"
        for split in SPLITS:
            img_dir = COCO_OUT / split / "images"
            if img_dir.exists():
                # Count originals of this source (exclude augments for clarity).
                # Bu kaynagin orijinallerini say (netlik icin augment haric).
                cnt = sum(
                    1 for p in img_dir.iterdir()
                    if not is_augmented(p.name) and source_of(p.name) == src
                )
            else:
                cnt = 0
            row += f"{cnt:>10}"
        print(row)


def main():
    print("=" * 70)
    print("  DATASET RE-STRATIFICATION")
    print("=" * 70)
    print(f"  Seed: {SEED}  Ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  COCO out: {COCO_OUT}")
    print(f"  YOLO out: {YOLO_OUT}")

    # Safety: refuse if output already exists.
    # Guvenlik: cikti zaten varsa reddet.
    if COCO_OUT.exists() or YOLO_OUT.exists():
        print("\n[ERROR] Output directory already exists. "
              "Remove it first or change the output path.")
        print(f"  rm -rf {COCO_OUT} {YOLO_OUT}")
        return

    # Step 1: collect files.
    # Adim 1: dosyalari topla.
    print("\n[1/6] Collecting all files...")
    originals, aug_by_original = collect_all_files()
    total_aug = sum(len(v) for v in aug_by_original.values())
    print(f"  Original frames: {len(originals)}")
    print(f"  Augmented files: {total_aug}")

    # Step 2: split originals.
    # Adim 2: orijinalleri bol.
    print("\n[2/6] Splitting originals (source-stratified)...")
    assignment = split_originals(originals)

    # Step 3: build per-split file lists.
    # Adim 3: split basina dosya listeleri olustur.
    print("\n[3/6] Building file lists...")
    split_files = build_file_lists(assignment, aug_by_original)
    for split in SPLITS:
        print(f"  {split}: {len(split_files[split])} total files")

    # Step 4: load COCO annotations.
    # Adim 4: COCO annotation'lari yukle.
    print("\n[4/6] Loading and merging COCO annotations...")
    categories, img_by_name, anns_by_name = load_all_coco_annotations()
    print(f"  Loaded metadata for {len(img_by_name)} images")

    # Step 5: write COCO + YOLO.
    # Adim 5: COCO + YOLO yaz.
    print("\n[5/6] Writing COCO splits (copying images)...")
    for split in SPLITS:
        write_coco_split(split, split_files[split],
                         categories, img_by_name, anns_by_name)

    print("\n[5/6] Writing YOLO splits (copying images + labels)...")
    for split in SPLITS:
        write_yolo_split(split, split_files[split])
    write_yolo_yaml()

    # Step 6: verify.
    # Adim 6: dogrula.
    print("\n[6/6] Verification...")
    verify_no_leakage(assignment, aug_by_original)
    report_source_distribution()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  New COCO: {COCO_OUT}")
    print(f"  New YOLO: {YOLO_OUT}")


if __name__ == "__main__":
    main()
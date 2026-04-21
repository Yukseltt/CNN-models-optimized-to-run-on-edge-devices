import json
import random
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")  # GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ──────────────────────────────────────────────
# KONFİGÜRASYON
# ──────────────────────────────────────────────
DATASET_ROOT = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/merged_thermal_coco_augmented"
SPLITS = ["train", "val", "test"]

# ──────────────────────────────────────────────
# YARDIMCI: COCO JSON YÜKLE
# ──────────────────────────────────────────────
def load_coco_json(split):
    ann_file = Path(DATASET_ROOT) / split / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"  [!] Annotation dosyası bulunamadı: {ann_file}")
        return None
    with open(ann_file, "r") as f:
        return json.load(f)

# ──────────────────────────────────────────────
# DOSYA SAYIMI
# ──────────────────────────────────────────────
def count_files_in_split(split):
    img_dir = Path(DATASET_ROOT) / split / "images"
    img_count = (
        len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        if img_dir.exists() else 0
    )
    coco = load_coco_json(split)
    if coco:
        registered = len(coco.get("images", []))
        ann_count  = len(coco.get("annotations", []))
        cat_count  = len(coco.get("categories", []))
        print(f"  {split.capitalize():6s} | "
              f"Klasörde: {img_count:6d} görüntü | "
              f"JSON'da: {registered:6d} görüntü | "
              f"Annotation: {ann_count:7d} | "
              f"Kategori: {cat_count:3d}")
    else:
        print(f"  {split.capitalize():6s} | Klasörde: {img_count:6d} görüntü | JSON yok")

# ──────────────────────────────────────────────
# ÖRNEK GÖRÜNTÜLERİ GÖSTER (bbox + kategori)
# ──────────────────────────────────────────────
def display_sample_images(split, num_samples=3):
    img_dir = Path(DATASET_ROOT) / split / "images"

    coco = load_coco_json(split)
    if coco is None:
        return

    categories     = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    valid_ids = list(img_id_to_anns.keys())
    if not valid_ids:
        print("  Annotation'lı görüntü bulunamadı.")
        return

    selected_ids = random.sample(valid_ids, min(num_samples, len(valid_ids)))

    fig, axes = plt.subplots(1, len(selected_ids), figsize=(6 * len(selected_ids), 6))
    if len(selected_ids) == 1:
        axes = [axes]

    for ax, img_id in zip(axes, selected_ids):
        info     = img_id_to_info[img_id]
        img_path = img_dir / info["file_name"]

        if not img_path.exists():
            ax.set_title(f"Bulunamadı:\n{info['file_name']}", fontsize=8)
            ax.axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)

        for ann in img_id_to_anns[img_id]:
            x, y, w, h = ann["bbox"]
            cat_name   = categories.get(ann["category_id"], "?")
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x, y - 4, cat_name, color="white", fontsize=7,
                    bbox=dict(facecolor="red", alpha=0.6, pad=1, edgecolor="none"))

        ax.set_title(info["file_name"], fontsize=8)
        ax.axis("off")

    plt.suptitle(f"{split.capitalize()} — Örnek Görüntüler", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────
# İSTATİSTİK YAZDIR
# ──────────────────────────────────────────────
def print_stats():
    print("=" * 70)
    print("COCO VERİ SETİ İSTATİSTİKLERİ")
    print("=" * 70)
    for split in SPLITS:
        count_files_in_split(split)

# ──────────────────────────────────────────────
# ANA AKIŞ (doğrudan çalıştırılınca)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print_stats()
    print("\nÖrnek görüntüler (bbox + kategori):")
    display_sample_images("train", num_samples=3)
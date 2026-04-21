import json

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# KONFİGÜRASYON
# ──────────────────────────────────────────────

# Kaynak COCO dataset klasörü
DATASET = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/dataset_augmented"

# Dönüştürülmüş YOLO dataset'inin kaydedileceği klasör
SAVE_DIR = "/home/atp-user-18/Desktop/uc_cihazlarda_terhmal_object_detection/dataset/dataset_augmented_yolo"

# Splitler
SPLITS = ["train", "val", "test"]

# Kategori isimleri (category_id → isim)
# COCO JSON'daki category id'lere göre düzenle
CATEGORY_NAMES = {
    0: "car",
    1: "person",
    2: "other_vehicle",
}


# ──────────────────────────────────────────────
# COCO → YOLO DÖNÜŞÜMÜ
# ──────────────────────────────────────────────

for split in ["train", "val", "test"]:
    json_path = f"{DATASET}/{split}/_annotations.coco.json"
    data = json.load(open(json_path))
    print(f"\n{split} kategorileri:")
    for cat in data["categories"]:
        print(f"  id: {cat['id']}  →  {cat['name']}")

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO formatı: [x_min, y_min, width, height]
    YOLO formatı: [x_center, y_center, width, height] (normalize edilmiş 0-1)
    """
    x, y, bw, bh = bbox
    x_center = (x + bw / 2) / img_w
    y_center  = (y + bh / 2) / img_h
    bw_norm   = bw / img_w
    bh_norm   = bh / img_h
    return x_center, y_center, bw_norm, bh_norm


def convert_split(split):
    json_path   = os.path.join(DATASET, split, "_annotations.coco.json")  # ✅
    img_src_dir = os.path.join(DATASET, split, "images")                   # ✅
    label_dir   = os.path.join(SAVE_DIR, "labels", split)
    img_dst_dir = os.path.join(SAVE_DIR, "images", split)

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(img_dst_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[WARN]  JSON bulunamadı, atlanıyor: {json_path}")
        return

    print(f"\n[INFO]  {split} split yükleniyor...")
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    print(f"[INFO]  {split} → {len(data['annotations'])} annotation işleniyor...")

    for ann in tqdm(data["annotations"], desc=f"  {split}"):
        img_id   = ann["image_id"]
        img_info = images[img_id]

        img_w     = img_info["width"]
        img_h     = img_info["height"]
        file_name = img_info["file_name"]

        x_center, y_center, bw, bh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        cls = ann["category_id"]

        label_filename = Path(file_name).stem + ".txt"
        label_path     = os.path.join(label_dir, label_filename)

        with open(label_path, "a") as f:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        src = os.path.join(img_src_dir, file_name)
        dst = os.path.join(img_dst_dir, file_name)

        if not os.path.exists(dst):
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"[WARN]  Görüntü bulunamadı: {src}")

    print(f"[OK]    {split} tamamlandı → {label_dir}")


# ──────────────────────────────────────────────
# YAML DOSYASI OLUŞTUR
# ──────────────────────────────────────────────

def create_yaml():
    """YOLO eğitimi için data.yaml dosyası oluşturur."""

    yaml_path = os.path.join(SAVE_DIR, "data.yaml")

    # Kategori isimlerini sıraya diz
    names_str = "\n".join(
        [f"  {idx}: {name}" for idx, name in sorted(CATEGORY_NAMES.items())]
    )

    yaml_content = f"""# Thermal Object Detection — YOLO Dataset
# Otomatik oluşturuldu: coco_to_yolo.py

path: {SAVE_DIR}

train: images/train
val:   images/val
test:  images/test

nc: {len(CATEGORY_NAMES)}

names:
{names_str}
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[OK]    YAML dosyası oluşturuldu: {yaml_path}")
    print(yaml_content)


# ──────────────────────────────────────────────
# İSTATİSTİK YAZDIR
# ──────────────────────────────────────────────

def print_stats():
    """Dönüştürülen dataset hakkında özet bilgi verir."""

    print("\n" + "=" * 60)
    print("DÖNÜŞÜM SONUCU İSTATİSTİKLER")
    print("=" * 60)

    for split in SPLITS:
        img_dir   = os.path.join(SAVE_DIR, "images", split)
        label_dir = os.path.join(SAVE_DIR, "labels", split)

        img_count   = len(list(Path(img_dir).glob("*.jpg"))) + \
                      len(list(Path(img_dir).glob("*.png"))) \
                      if os.path.exists(img_dir) else 0

        label_count = len(list(Path(label_dir).glob("*.txt"))) \
                      if os.path.exists(label_dir) else 0

        print(f"  {split.capitalize():6s} | "
              f"Görüntü: {img_count:6d} | "
              f"Label: {label_count:6d}")

    print("=" * 60)


# ──────────────────────────────────────────────
# ANA AKIŞ
# ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("COCO → YOLO FORMAT DÖNÜŞÜMÜ")
    print("=" * 60)
    print(f"  Kaynak  : {DATASET}")
    print(f"  Hedef   : {SAVE_DIR}")
    print(f"  Splitler: {SPLITS}")
    print(f"  Sınıflar: {CATEGORY_NAMES}")
    print("=" * 60)

    # Her split için dönüşüm yap
    for split in SPLITS:
        convert_split(split)

    # YAML dosyası oluştur
    create_yaml()

    # İstatistikleri yazdır
    print_stats()

    print("\n[OK]    Tüm dönüşümler tamamlandı!")
    print(f"[INFO]  YAML yolu: {SAVE_DIR}/data.yaml")
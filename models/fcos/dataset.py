# dataset.py
#
# Custom COCO Dataset for Torchvision FCOS (FCOS-adapted).
# Torchvision FCOS (FCOS-adapted) icin custom COCO Dataset.
#
# Reads a single _annotations.coco.json file plus an images/ folder.
# Tek bir _annotations.coco.json dosyasini ve images/ klasorunu okur.
#
# Returns (image_tensor, target_dict) per Torchvision's FCOS contract
# (same contract as FCOS: list of images + list of target dicts).
# Torchvision FCOS (FCOS-adapted) beklentisine gore (image_tensor, target_dict) dondurur
# (FCOS ile ayni: goruntu listesi + target dict listesi).
#
# FCOS has an internal GeneralizedRCNNTransform-like resizer/normalizer,
# so we feed raw images at original resolution in [0, 1] range; the model
# rescales internally to 300x300 (FCOS300) or 320x320 (FCOSLite).
# FCOS'nin kendi icinde GeneralizedRCNNTransform benzeri resize/normalize
# var, bu yuzden orijinal cozunurlukte [0, 1] araliginda raw goruntuler
# veriyoruz; model dahili olarak 300x300 (FCOS300) veya 320x320 (FCOSLite)
# boyutuna olceklendiriyor.
#
# Target dict format / Target dict formati:
#     {
#         "boxes":    Tensor[N, 4]   in [x_min, y_min, x_max, y_max] pixel format
#         "labels":   Tensor[N]      class id per box
#         "image_id": Tensor[1]      unique image id
#         "area":     Tensor[N]      area of each box
#         "iscrowd":  Tensor[N]      0 or 1 crowd flag
#     }

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# Helpers / Yardimcilar.

def build_train_transform() -> A.Compose:
    # Minimal augmentation matching the YOLO setup.
    # YOLO setup'i ile esit minimum augmentation.
    #
    # Dataset is already pre-augmented (2x_augmented_*), so we keep this light.
    # Dataset zaten augmente edilmis (2x_augmented_*), bu yuzden hafif tutuyoruz.
    return A.Compose(
        [
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7  * 255),
                val_shift_limit=int(0.4  * 255),
                p=0.5,
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def build_eval_transform() -> A.Compose:
    # No augmentation for val / test, only normalize and tensorify.
    # Val / test icin augmentation yok, sadece normalize ve tensor donusumu.
    return A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
        ),
    )


# Dataset class.

class CocoDetectionDataset(Dataset):
    # Custom COCO dataset that returns FCOS compatible targets.
    # FCOS uyumlu target dondurur custom COCO dataset.
    #
    # Args / Parametreler:
    #     images_dir: folder containing the .jpg / .png files.
    #     images_dir: .jpg / .png dosyalarini iceren klasor.
    #
    #     annotation_path: path to _annotations.coco.json.
    #     annotation_path: _annotations.coco.json dosya yolu.
    #
    #     transform: optional Albumentations Compose with bbox_params.
    #     transform: opsiyonel Albumentations Compose, bbox_params ile.

    def __init__(
        self,
        images_dir: str,
        annotation_path: str,
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transform  = transform

        # Load JSON once during init.
        # JSON'i init sirasinda bir kez yukle.
        with open(annotation_path, "r") as f:
            coco = json.load(f)

        # Index annotations by image_id for fast lookup in __getitem__.
        # __getitem__'da hizli erisim icin annotations'i image_id ile indeksle.
        self.images = coco["images"]
        self.anns_by_image_id: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image_id[ann["image_id"]].append(ann)

        # Build category id remap so labels are always contiguous and 1-based.
        # Etiketler her zaman ardisik ve 1'den baslayacak sekilde category id remap olustur.
        #
        # Torchvision FCOS (FCOS-adapted) reserves label 0 for background.
        # Torchvision FCOS (FCOS-adapted), 0 etiketini background icin ayirir.
        # COCO category ids may start at 0 (as in our dataset), so we shift +1.
        # COCO category id'leri 0'dan baslayabilir (bizim datasetimizdeki gibi),
        # bu yuzden +1 kaydiriyoruz.
        sorted_cats = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat_id_to_label: dict[int, int] = {}
        self.class_names: list[str] = []
        for i, cat in enumerate(sorted_cats, start=1):
            self.cat_id_to_label[cat["id"]] = i
            self.class_names.append(cat["name"])
        self.num_classes = len(self.class_names) + 1  # +1 for background

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        image_id = img_info["id"]

        # Read image with OpenCV (BGR by default), convert to RGB.
        # OpenCV ile goruntuyu oku (varsayilan BGR), RGB'ye cevir.
        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not readable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image size for bbox clipping.
        # Bbox kirpma icin goruntu boyutu.
        img_h, img_w = image.shape[:2]

        anns = self.anns_by_image_id.get(image_id, [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Skip degenerate boxes that have zero width or height.
            # Genisligi veya yuksekligi sifir olan bozuk bbox'leri atla.
            if w <= 0 or h <= 0:
                continue
            # Convert to pascal_voc and clip to image bounds.
            # pascal_voc'a cevir ve goruntu sinirlarina kirp.
            x_min = max(0.0,         float(x))
            y_min = max(0.0,         float(y))
            x_max = min(float(img_w), float(x + w))
            y_max = min(float(img_h), float(y + h))
            # Skip if clipping made the box degenerate.
            # Kirpma sonrasi bbox bozulduysa atla.
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", (x_max - x_min) * (y_max - y_min))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        # Apply Albumentations transform if provided.
        # Albumentations transform varsa uygula.
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels,
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
            # Areas and iscrowd lists may be misaligned after augmentation drops boxes.
            # Augmentation bbox dustugunde areas ve iscrowd listeleri hizalanmaz.
            # Recompute area from the surviving boxes for safety.
            # Guvenlik icin alanlari hayatta kalan bbox'lerden tekrar hesapla.
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            iscrowd = [0] * len(boxes)
        else:
            # No transform: convert image to tensor manually.
            # Transform yoksa goruntuyu manuel olarak tensor'a cevir.
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert lists to tensors with the right dtypes for Torchvision.
        # Listeleri Torchvision'in bekledigi dtype'larla tensor'a cevir.
        if len(boxes) == 0:
            boxes_t   = torch.zeros((0, 4), dtype=torch.float32)
            labels_t  = torch.zeros((0,),    dtype=torch.int64)
            areas_t   = torch.zeros((0,),    dtype=torch.float32)
            iscrowd_t = torch.zeros((0,),    dtype=torch.int64)
        else:
            boxes_t   = torch.as_tensor(boxes,   dtype=torch.float32)
            labels_t  = torch.as_tensor(labels,  dtype=torch.int64)
            areas_t   = torch.as_tensor(areas,   dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     areas_t,
            "iscrowd":  iscrowd_t,
        }

        return image, target


# Collate function / Collate fonksiyonu.

def collate_fn(batch):
    # Default DataLoader collate stacks tensors which fails for variable-size targets.
    # Varsayilan DataLoader collate tensorleri stack eder, degisken boyutlu target'larda hata verir.
    #
    # Torchvision detection models expect a list of tensors and a list of dicts.
    # Torchvision detection modelleri tensor listesi ve dict listesi bekler.
    return tuple(zip(*batch))

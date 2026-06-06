"""Lokal COCO formatli termal dataset wrapper'lari.

Colab notebook'tan ayiklandi; HuggingFace Trainer + DETR image processor
icin uygun batch'i uretir.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchBaseDataset


@dataclass
class _SplitWrapper:
    images_dir: Path
    images: list
    anns_by_img: dict
    category_names: list


class LocalCocoDataset(TorchBaseDataset):
    """Tek COCO split icin minimal wrapper.

    `_annotations.coco.json` + `images/` yapisini bekler.
    HuggingFace CPPE-5 ornegiyle ayni dict shape'i doner.
    """

    def __init__(self, split_dir: str | Path):
        split_dir = Path(split_dir)
        self.images_dir = split_dir / "images"
        with open(split_dir / "_annotations.coco.json") as f:
            data = json.load(f)

        self.categories = sorted(data["categories"], key=lambda c: c["id"])
        self.images     = data["images"]

        self._anns_by_img: dict = {}
        for a in data["annotations"]:
            self._anns_by_img.setdefault(a["image_id"], []).append(a)

    @property
    def category_names(self) -> list[str]:
        return [c["name"] for c in self.categories]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        info = self.images[idx]
        anns = self._anns_by_img.get(info["id"], [])

        image_path = self.images_dir / info["file_name"]
        image = Image.open(image_path).convert("RGB")

        bboxes     = [a["bbox"] for a in anns]
        categories = [a["category_id"] for a in anns]
        areas      = [a.get("area", a["bbox"][2] * a["bbox"][3]) for a in anns]
        ids        = [a["id"] for a in anns]

        return {
            "image_id": info["id"],
            "image":    image,
            "width":    info["width"],
            "height":   info["height"],
            "objects":  {
                "id":       ids,
                "bbox":     bboxes,
                "category": categories,
                "area":     areas,
            },
        }


class ThermalCocoDataset(TorchBaseDataset):
    """Image processor + augmentation ile sarmalanmis COCO dataset.

    Hugging Face Trainer'in bekledigi format'i (pixel_values + labels) doner.
    """

    def __init__(self, base: LocalCocoDataset, image_processor, transform=None):
        self.base            = base
        self.image_processor = image_processor
        self.transform       = transform

    @staticmethod
    def _format_anns(image_id, categories, boxes):
        return {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id":    image_id,
                    "category_id": int(cat),
                    "bbox":        list(box),
                    "iscrowd":     0,
                    "area":        float(box[2] * box[3]),
                }
                for cat, box in zip(categories, boxes)
            ],
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample     = self.base[idx]
        image_id   = sample["image_id"]
        image      = np.array(sample["image"])
        boxes      = sample["objects"]["bbox"]
        categories = sample["objects"]["category"]

        if self.transform is not None:
            out = self.transform(image=image, bboxes=boxes, category=categories)
            image      = out["image"]
            boxes      = out["bboxes"]
            categories = out["category"]

        anns = self._format_anns(image_id, categories, boxes)
        encoding = self.image_processor(
            images=image, annotations=anns, return_tensors="pt"
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels":       encoding["labels"][0],
        }


def collate_fn(batch: list) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       [b["labels"] for b in batch],
    }


def build_transforms():
    """Train ve val icin albumentations transformlari."""
    import albumentations as A

    train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"],
            clip=True, min_area=25, min_width=1, min_height=1,
        ),
    )
    val = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"],
            clip=True, min_area=1, min_width=1, min_height=1,
        ),
    )
    return train, val

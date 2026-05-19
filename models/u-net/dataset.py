# dataset.py
#
# U-Net image+mask dataset for thermal segmentation.
# Termal segmentasyon icin U-Net image+mask dataset'i.
#
# Beklenen klasor yapisi (coco_to_unet_sam2.ipynb cikti formati):
#   <data_dir>/<split>/images/*.jpg
#   <data_dir>/<split>/masks/*.png    # P-mode, pixel = class_id (0=bg, 1..N)
#
# Image-mask eslestirmesi dosya adina gore (uzanti haric).

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".png", ".bmp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _list_stems(folder: Path, exts) -> List[str]:
    return sorted(p.stem for p in folder.iterdir() if p.suffix.lower() in exts)


def _find_with_ext(folder: Path, stem: str, exts) -> Optional[Path]:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


class UNetSegDataset(Dataset):
    # Image + mask cifti dondurur.
    # image: float tensor (C, H, W), normalize edilmis
    # mask:  long tensor (H, W), pixel degerleri sinif index'leri

    def __init__(
        self,
        images_dir: str,
        masks_dir:  str,
        input_size: int  = 512,
        augment:    bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.input_size = int(input_size)
        self.augment    = augment

        img_stems  = set(_list_stems(self.images_dir, IMG_EXTS))
        mask_stems = set(_list_stems(self.masks_dir,  MASK_EXTS))
        common     = sorted(img_stems & mask_stems)

        if not common:
            raise RuntimeError(
                f"No matching image/mask pairs found.\n"
                f"  images_dir: {self.images_dir} ({len(img_stems)} files)\n"
                f"  masks_dir:  {self.masks_dir} ({len(mask_stems)} files)"
            )

        self.samples: List[Tuple[Path, Path]] = []
        for stem in common:
            ip = _find_with_ext(self.images_dir, stem, IMG_EXTS)
            mp = _find_with_ext(self.masks_dir,  stem, MASK_EXTS)
            if ip is not None and mp is not None:
                self.samples.append((ip, mp))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if img.size != (self.input_size, self.input_size):
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        if mask.size != (self.input_size, self.input_size):
            mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

        img_arr  = np.asarray(img,  dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.int64)

        if self.augment:
            if np.random.rand() < 0.5:
                img_arr  = img_arr[:,  ::-1, :].copy()
                mask_arr = mask_arr[:, ::-1   ].copy()

        img_arr = (img_arr - IMAGENET_MEAN) / IMAGENET_STD
        img_arr = np.transpose(img_arr, (2, 0, 1))

        img_t  = torch.from_numpy(img_arr).float()
        mask_t = torch.from_numpy(mask_arr).long()
        return img_t, mask_t


def compute_class_pixel_counts(dataset: UNetSegDataset, num_classes: int) -> np.ndarray:
    # Egitim seti uzerinden sinif basina toplam piksel sayisi.
    # Sinif-agirlikli loss icin kullanilir.
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, mask_path in dataset.samples:
        m   = np.asarray(Image.open(mask_path), dtype=np.int64)
        bc  = np.bincount(m.ravel(), minlength=num_classes)
        counts[: len(bc)] += bc[: num_classes]
    return counts

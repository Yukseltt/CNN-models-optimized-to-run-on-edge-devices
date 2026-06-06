# hf_detector.py
#
# Thin adapter that makes a HuggingFace object-detection model (RT-DETR,
# DETR, ...) behave exactly like a Torchvision detection model.
# Bir HuggingFace object-detection modelini (RT-DETR, DETR, ...) tamamen
# bir Torchvision detection modeli gibi davranmaya zorlayan ince adapter.
#
# Why / Neden:
#     The V3 benchmark runners speak two dialects: "yolo" (Ultralytics) and
#     everything else (Torchvision-style: model(list_of_CHW_float01_tensors)
#     returns a list of dicts with "boxes" (xyxy abs), "labels" (1-based) and
#     "scores"). By wrapping the HF model in that exact contract, RT-DETR and
#     DETR flow through the existing Faster R-CNN accuracy / speed / memory /
#     determinism code with zero changes elsewhere.
#     V3 benchmark runner'lari iki lehce konusur: "yolo" (Ultralytics) ve
#     digerleri (Torchvision tarzi: model(CHW_float01_tensor_listesi) ->
#     "boxes" (xyxy abs), "labels" (1-based) ve "scores" iceren dict listesi
#     doner). HF modelini tam bu kontrata sararak RT-DETR ve DETR mevcut
#     Faster R-CNN dogruluk / hiz / bellek / determinizm koduna hicbir
#     degisiklik olmadan akar.

import numpy as np
import torch
import torch.nn as nn


class HFDetectorWrapper(nn.Module):
    # Wrap a HF AutoModelForObjectDetection + image processor.
    # Bir HF AutoModelForObjectDetection + image processor sarar.

    def __init__(self, model, image_processor, threshold: float = 0.001):
        super().__init__()
        self.model           = model
        self.image_processor = image_processor
        # Low threshold so COCOeval sees the full precision-recall curve,
        # matching the conf=0.001 protocol used for YOLO accuracy.
        # Dusuk threshold ile COCOeval tum precision-recall egrisini gorur,
        # YOLO dogrulugundaki conf=0.001 protokolu ile ayni.
        self.threshold       = threshold

    @torch.no_grad()
    def forward(self, images: list) -> list:
        # images: list of CHW float[0,1] RGB tensors (Torchvision contract).
        # images: CHW float[0,1] RGB tensor listesi (Torchvision kontrati).
        device = next(self.model.parameters()).device

        # The HF image processor re-applies its own rescale + normalize, so
        # we hand it plain uint8 HWC RGB arrays (undo the [0,1] scaling).
        # HF image processor kendi rescale + normalize'ini tekrar uygular,
        # bu yuzden ona duz uint8 HWC RGB array veriyoruz ([0,1] olcegini geri al).
        np_imgs      = []
        target_sizes = []
        for img in images:
            _, h, w = img.shape
            target_sizes.append((h, w))
            arr = (img.detach().cpu().clamp(0, 1).mul(255).round().byte()
                   .permute(1, 2, 0).numpy().astype(np.uint8))
            np_imgs.append(arr)

        enc = self.image_processor(images=np_imgs, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)

        outputs = self.model(pixel_values=pixel_values)

        ts = torch.tensor(target_sizes, device=device)
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=ts,
        )

        # Convert HF 0-based labels to the 1-based convention used by the
        # Faster R-CNN runner (which maps category_id = label - 1).
        # HF 0-based label'lari Faster R-CNN runner'inin kullandigi 1-based
        # konvansiyona cevir (runner category_id = label - 1 yapar).
        formatted = []
        for r in results:
            formatted.append({
                "boxes":  r["boxes"],
                "labels": r["labels"] + 1,
                "scores": r["scores"],
            })
        return formatted

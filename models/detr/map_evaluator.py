"""DETR icin compute_metrics fonksiyonu (torchmetrics MeanAveragePrecision).

Colab notebook'tan ayiklandi. HuggingFace Trainer'a verilebilen
`eval_compute_metrics_fn` callable'i uretir.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format


@dataclass
class _Out:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    def __init__(self, image_processor, threshold: float = 0.0, id2label=None):
        self.image_processor = image_processor
        self.threshold       = threshold
        self.id2label        = id2label

    @staticmethod
    def _image_sizes(targets):
        sizes = []
        for batch in targets:
            sizes.append(torch.tensor(np.array([x["size"] for x in batch])))
        return sizes

    def _collect_targets(self, targets, image_sizes):
        post = []
        for tgt_batch, size_batch in zip(targets, image_sizes):
            for target, size in zip(tgt_batch, size_batch):
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                h, w = size
                boxes = boxes * torch.tensor([w, h, w, h])
                labels = torch.tensor(target["class_labels"])
                post.append({"boxes": boxes, "labels": labels})
        return post

    def _collect_preds(self, predictions, image_sizes):
        post = []
        for pred_batch, size_batch in zip(predictions, image_sizes):
            batch_logits, batch_boxes = pred_batch[1], pred_batch[2]
            out = _Out(
                logits=torch.tensor(batch_logits),
                pred_boxes=torch.tensor(batch_boxes),
            )
            results = self.image_processor.post_process_object_detection(
                out, threshold=self.threshold, target_sizes=size_batch
            )
            post.extend(results)
        return post

    @torch.no_grad()
    def __call__(self, eval_pred):
        predictions, targets = eval_pred.predictions, eval_pred.label_ids
        image_sizes = self._image_sizes(targets)
        post_preds   = self._collect_preds(predictions, image_sizes)
        post_targets = self._collect_targets(targets, image_sizes)

        evaluator = MeanAveragePrecision(
            box_format="xyxy", class_metrics=True,
        )
        evaluator.warn_on_many_detections = False
        evaluator.update(post_preds, post_targets)
        metrics = evaluator.compute()

        classes = metrics.pop("classes")
        map_per_class    = metrics.pop("map_per_class")
        mar100_per_class = metrics.pop("mar_100_per_class")
        for cls_id, m, r in zip(classes, map_per_class, mar100_per_class):
            name = (
                self.id2label[int(cls_id.item())]
                if self.id2label and int(cls_id.item()) in self.id2label
                else f"class_{int(cls_id.item())}"
            )
            metrics[f"map_{name}"]     = float(m.item())
            metrics[f"mar_100_{name}"] = float(r.item())

        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        return metrics

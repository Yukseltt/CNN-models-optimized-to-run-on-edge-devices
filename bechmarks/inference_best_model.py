"""
Inference with the best benchmarked model (faster_rcnn_resnet50_28.04.2026).
30 rastgele test goruntusu uzerinde inference + tahmin gorseli kaydeder.
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "faster_rcnn"))

from model_factory import create_faster_rcnn


CKPT_PATH    = PROJECT_ROOT / "runs" / "faster_rcnn_resnet50_28.04.2026" / "best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "dataset" / "2x_augmented_coco_dataset" / "dataset_augmented" / "test" / "images"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "inference_faster_rcnn_resnet50_top30"

NUM_IMAGES = 30
SCORE_THRESHOLD = 0.5
SEED = 42

CLASS_COLORS = {
    "Person":       (0, 255, 0),
    "Car":          (255, 0, 0),
    "OtherVehicle": (0, 165, 255),
}


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = len(class_names) + 1  # +1 background
    backbone    = ckpt["cfg"].get("BACKBONE", "resnet50")

    model = create_faster_rcnn(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=None,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, class_names, ckpt.get("metric"), ckpt.get("epoch")


def preprocess(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def draw_predictions(image_bgr, boxes, labels, scores, class_names):
    out = image_bgr.copy()
    for box, label_id, score in zip(boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        cname = class_names[label_id - 1] if 1 <= label_id <= len(class_names) else f"id{label_id}"
        color = CLASS_COLORS.get(cname, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{cname} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {CKPT_PATH}")
    model, class_names, metric, epoch = load_model(CKPT_PATH, device)
    print(f"  classes: {class_names}  ckpt mAP@0.5={metric}  epoch={epoch}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    all_images = sorted(TEST_IMG_DIR.glob("*.jpg"))
    selected = random.sample(all_images, NUM_IMAGES)
    print(f"Selected {len(selected)} of {len(all_images)} test images")

    counts = {n: 0 for n in class_names}

    with torch.inference_mode():
        for i, img_path in enumerate(selected, start=1):
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print(f"  [{i}] SKIP unreadable: {img_path.name}")
                continue
            tensor = preprocess(image_bgr, device)
            out = model([tensor])[0]
            boxes  = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            kept = scores >= SCORE_THRESHOLD
            for lab in labels[kept]:
                if 1 <= lab <= len(class_names):
                    counts[class_names[lab - 1]] += 1

            vis = draw_predictions(image_bgr, boxes, labels, scores, class_names)
            out_path = OUTPUT_DIR / f"pred_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  [{i:02d}/{NUM_IMAGES}] {img_path.name} -> {int(kept.sum())} det")

    print("\nDetections (score >= {:.2f}):".format(SCORE_THRESHOLD))
    for n, c in counts.items():
        print(f"  {n:15s}: {c}")
    print(f"\nOutput dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

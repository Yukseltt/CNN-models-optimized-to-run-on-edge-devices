# memory_profiler.py
#
# Model profiling: parameter count, FLOPs, GPU memory, disk size.
# Model profilleme: parametre sayisi, FLOPs, GPU bellek, disk boyutu.

import os
from pathlib import Path

import torch
import torch.nn as nn


def profile_params(model, family: str) -> dict:
    # Count total and trainable parameters.
    # Toplam ve egitilebilir parametreleri say.
    if family == "yolo":
        pytorch_model = model.model
    else:
        pytorch_model = model

    total = sum(p.numel() for p in pytorch_model.parameters())
    trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)

    return {
        "params_total_M":     round(total / 1e6, 2),
        "params_trainable_M": round(trainable / 1e6, 2),
    }


def profile_flops(model, family: str, device: str = "cuda:0") -> dict:
    # Estimate FLOPs using thop library.
    # thop kutuphanesi ile FLOPs tahmin et.
    #
    # Falls back gracefully if thop is unavailable or fails.
    # thop mevcut degilse veya basarisiz olursa nazikce geri doner.
    try:
        from thop import profile as thop_profile
    except ImportError:
        return {"flops_G": None}

    try:
        dummy = torch.randn(1, 3, 640, 640).to(device)

        if family == "yolo":
            pytorch_model = model.model
            flops, _ = thop_profile(pytorch_model, inputs=(dummy,), verbose=False)
        else:
            # Faster R-CNN expects a list of tensors.
            # Faster R-CNN tensor listesi bekler.
            flops, _ = thop_profile(model, inputs=([dummy],), verbose=False)

        return {"flops_G": round(flops / 1e9, 2)}
    except Exception:
        return {"flops_G": None}


def profile_gpu_memory(model, family: str, device: str = "cuda:0") -> dict:
    # Measure peak GPU memory during a single inference.
    # Tek bir inference sirasinda peak GPU bellegini olc.
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize()

    dummy = torch.randn(1, 3, 640, 640).to(device)

    if family == "yolo":
        model.predict(
            source=dummy.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            conf=0.25, save=False, verbose=False,
        )
    else:
        model.eval()
        with torch.no_grad():
            model([dummy.squeeze(0)])

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device=device)

    return {"gpu_mem_peak_MB": round(peak_bytes / (1024 * 1024), 1)}


def profile_disk_size(best_pt: Path) -> dict:
    # Get best.pt file size in megabytes.
    # best.pt dosya boyutunu megabyte olarak al.
    size_bytes = os.path.getsize(best_pt)
    return {"disk_size_MB": round(size_bytes / (1024 * 1024), 2)}


def run_memory_profile(model, family: str, best_pt: Path, device: str = "cuda:0") -> dict:
    # Run all profiling measurements and combine results.
    # Tum profilleme olcumlerini calistir ve sonuclari birlestir.
    result = {}
    result.update(profile_params(model, family))
    result.update(profile_flops(model, family, device))
    result.update(profile_gpu_memory(model, family, device))
    result.update(profile_disk_size(best_pt))
    return result
# benchmark.py
#
# V3 benchmark orchestrator for all detection models.
# Tum detection modelleri icin V3 benchmark orkestrator.
#
# Runs accuracy, speed, memory profiling, and determinism checks
# on every model in runs/ and writes results to a 4-sheet Excel.
# runs/ altindaki her model uzerinde dogruluk, hiz, bellek profili ve
# determinizm kontrolleri calistirir ve sonuclari 4 sayfali Excel'e yazar.
#
# Usage / Kullanim:
#     python -m src.model_benchmark.benchmark
#     python -m src.model_benchmark.benchmark --runs-dir /path/to/runs
#     python -m src.model_benchmark.benchmark --output results/benchmark.xlsx

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO

# Ensure project root is in path.
# Proje kokunu path'te oldugundan emin ol.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_benchmark.model_loaders import load_model, discover_runs
from src.model_benchmark.accuracy_runner import run_yolo_accuracy, run_frcnn_accuracy
from src.model_benchmark.speed_runner import run_yolo_speed, run_frcnn_speed
from src.model_benchmark.memory_profiler import run_memory_profile
from src.model_benchmark.determinism_checker import check_determinism
from src.model_benchmark.excel_writer import write_benchmark_excel


# Default paths / Varsayilan yollar.

DEFAULT_RUNS_DIR    = PROJECT_ROOT / "runs"
DEFAULT_TEST_DIR    = PROJECT_ROOT / "dataset" / "2x_augmented_coco_dataset" / "dataset_augmented" / "test"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "benchmark_v3.xlsx"


def _clear_gpu():
    # Free GPU memory between models.
    # Modeller arasi GPU bellegini serbest birak.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_single(
    run_dir:         Path,
    test_dir:        Path,
    test_images_dir: str,
    coco_gt:         COCO,
    device:          str,
) -> dict:
    # Run the full V3 benchmark on a single model.
    # Tek bir model uzerinde tam V3 benchmark calistir.
    print(f"\n{'='*70}")
    print(f"  Model: {run_dir.name}")
    print(f"{'='*70}")

    _clear_gpu()

    # Load model.
    # Modeli yukle.
    t0 = time.time()
    info = load_model(run_dir, device=device)
    if info is None:
        print(f"  SKIP: best.pt not found")
        return None
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s  |  family={info['family']}")

    model  = info["model"]
    family = info["family"]
    result = {
        "run_name":    info["run_name"],
        "family":      family,
        "class_names": info.get("class_names", []),
    }

    # 1. Accuracy (mAP on test set).
    # 1. Dogruluk (test set uzerinde mAP).
    print(f"  [1/4] Accuracy (test set, conf=0.001)...")
    t0 = time.time()
    try:
        if family == "yolo":
            acc = run_yolo_accuracy(model, test_images_dir, coco_gt)
        else:
            acc = run_frcnn_accuracy(model, str(test_dir), coco_gt, device)
        result["map50"]     = acc["map50"]
        result["map50_95"]  = acc["map50_95"]
        result["per_class"] = acc["per_class"]
        print(f"         mAP@0.5={acc['map50']:.4f}  mAP@0.5:0.95={acc['map50_95']:.4f}")
    except Exception as e:
        print(f"         ERROR: {e}")
        result["map50"] = None
        result["map50_95"] = None
        result["per_class"] = {}
    print(f"         Done in {time.time() - t0:.1f}s")

    # 2. Speed (latency, batch=1, conf=0.25).
    # 2. Hiz (gecikme, batch=1, conf=0.25).
    print(f"  [2/4] Speed (batch=1, conf=0.25, 20w+100m)...")
    t0 = time.time()
    try:
        if family == "yolo":
            spd = run_yolo_speed(model, test_images_dir)
        else:
            spd = run_frcnn_speed(model, test_images_dir, device)
        result.update(spd)
        print(f"         latency={spd['latency_mean_ms']:.2f}ms  FPS={spd['fps']:.1f}")
    except Exception as e:
        print(f"         ERROR: {e}")
        result["latency_mean_ms"] = None
        result["fps"] = None
    print(f"         Done in {time.time() - t0:.1f}s")

    # 3. Memory and size profiling.
    # 3. Bellek ve boyut profili.
    print(f"  [3/4] Memory & size profiling...")
    t0 = time.time()
    try:
        mem = run_memory_profile(model, family, info["best_pt"], device)
        result.update(mem)
        print(f"         params={mem['params_total_M']:.2f}M  "
              f"disk={mem['disk_size_MB']:.2f}MB  "
              f"gpu_peak={mem['gpu_mem_peak_MB']:.1f}MB")
    except Exception as e:
        print(f"         ERROR: {e}")
    print(f"         Done in {time.time() - t0:.1f}s")

    # 4. Determinism check.
    # 4. Determinizm kontrolu.
    print(f"  [4/4] Determinism check...")
    t0 = time.time()
    try:
        det = check_determinism(model, family, test_images_dir, device)
        result.update(det)
        status = "PASS" if det["deterministic"] else "FAIL"
        print(f"         {status}  max_diff={det['max_diff']}")
    except Exception as e:
        print(f"         ERROR: {e}")
        result["deterministic"] = None
    print(f"         Done in {time.time() - t0:.1f}s")

    # Cleanup.
    # Temizlik.
    del model
    if "model" in info:
        del info["model"]
    _clear_gpu()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="V3 benchmark: accuracy + speed + memory + determinism"
    )
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                        help="Path to runs directory")
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR),
                        help="Path to COCO format test directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH),
                        help="Output Excel path")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--skip", nargs="*", default=None,
                        help="Additional directory name patterns to skip")
    parser.add_argument("--include", nargs="*", default=None,
                        help="If set, only benchmark these exact directory names")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    test_dir = Path(args.test_dir)
    test_images_dir = str(test_dir / "images")
    output_path = Path(args.output)

    skip_patterns = [
        "detect", "efficientnet", "unet",
        "sayzek_runs_yolo11_base_param_puring",
        "sayzek_runs_yolo11_base_param",
        "thermal_yolo",
        "yolo11n_baseline_no_cb",
        "yolo26n_default_augmented_2026_05_14",
    ]
    # Keep yolo26n_default_augmented_2026_05_14-2, skip the one without -2.
    # -2 olmayani atla, -2 olani tut.

    if args.skip:
        skip_patterns.extend(args.skip)

    print(f"[Benchmark] Runs dir:    {runs_dir}")
    print(f"[Benchmark] Test dir:    {test_dir}")
    print(f"[Benchmark] Output:      {output_path}")
    print(f"[Benchmark] Device:      {args.device}")
    print(f"[Benchmark] Skip:        {skip_patterns}")

    # Validate test set.
    # Test setini dogrula.
    ann_path = test_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"ERROR: Test annotation not found: {ann_path}")
        sys.exit(1)

    print(f"\n[Benchmark] Loading test set annotations...")
    coco_gt = COCO(str(ann_path))
    n_images = len(coco_gt.getImgIds())
    n_anns = len(coco_gt.getAnnIds())
    print(f"[Benchmark] Test set: {n_images} images, {n_anns} annotations")

    # Discover models.
    # Modelleri bul.
    run_dirs = discover_runs(runs_dir, skip_patterns)

    # If --include is set, keep only exact matches.
    # --include verilmisse sadece tam eslesen dizinleri tut.
    if args.include:
        run_dirs = [d for d in run_dirs if d.name in args.include]

    print(f"\n[Benchmark] Found {len(run_dirs)} models:")
    for d in run_dirs:
        print(f"  - {d.name}")

    # Run benchmarks.
    # Benchmark'lari calistir.
    all_results = []
    total_start = time.time()

    for run_dir in run_dirs:
        result = benchmark_single(
            run_dir=run_dir,
            test_dir=test_dir,
            test_images_dir=test_images_dir,
            coco_gt=coco_gt,
            device=args.device,
        )
        if result is not None:
            all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"[Benchmark] Completed {len(all_results)} models in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Write results.
    # Sonuclari yaz.
    write_benchmark_excel(all_results, output_path)

    # Print top 5 summary.
    # Ilk 5 ozetini yazdir.
    print(f"\n[Benchmark] Top 5 by mAP@0.5:")
    sorted_r = sorted(all_results, key=lambda x: x.get("map50", 0) or 0, reverse=True)
    for i, r in enumerate(sorted_r[:5]):
        print(f"  {i+1}. {r['run_name']:<55s}  "
              f"mAP@0.5={r.get('map50', 0):.4f}  "
              f"latency={r.get('latency_mean_ms', 0):.1f}ms")


if __name__ == "__main__":
    main()
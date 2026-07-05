# bench_core.py
#
# Cihaz-bagimsiz olcum cekirdegi (src/edge_benchmark/bench_core.py'den uyarlandi)
# + edge_perf'e ozel SIRALAMA (ranking) ve birlesik rapor.
#
#   - latency_stats        : gecikme mean/median/p90/p95/p99/std/min/max + FPS
#   - BackgroundSampler    : guc/sicaklik/util arka plan ornekleyici
#   - evaluate_map         : pycocotools mAP@0.5 / mAP@0.5:0.95 + per-class AP@0.5
#   - letterbox/list_images: on-isleme yardimcilari
#   - derive_efficiency    : FPS/W, mJ/frame, mAP/W, mAP/Mparam
#   - rank_models          : modelleri secilen metriklere gore sirala
#   - save_results         : JSON + CSV + sirali leaderboard yazar/basar

import csv
import json
import os
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# =====================================================================
# Gecikme istatistikleri
# =====================================================================

def latency_stats(times_ms: list) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return {}
    mean = float(arr.mean())

    def pct(p):
        return float(np.percentile(arr, p))

    return {
        "latency_mean_ms":   round(mean, 3),
        "latency_median_ms": round(float(np.median(arr)), 3),
        "latency_p90_ms":    round(pct(90), 3),
        "latency_p95_ms":    round(pct(95), 3),
        "latency_p99_ms":    round(pct(99), 3),
        "latency_std_ms":    round(float(arr.std()), 3),
        "latency_min_ms":    round(float(arr.min()), 3),
        "latency_max_ms":    round(float(arr.max()), 3),
        "fps_mean":          round(1000.0 / mean, 2) if mean > 0 else 0.0,
        "n":                 int(arr.size),
    }


# =====================================================================
# Arka plan ornekleyici (guc / sicaklik / util)
# =====================================================================

class BackgroundSampler:
    def __init__(self, sample_fn, interval: float = 0.1, name: str = "sample"):
        self.sample_fn = sample_fn
        self.interval = interval
        self.name = name
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                v = self.sample_fn()
                if v is not None:
                    self._samples.append(float(v))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self._samples:
            return {f"{self.name}_avg": None, f"{self.name}_peak": None,
                    f"{self.name}_min": None, f"{self.name}_n": 0}
        s = self._samples
        return {
            f"{self.name}_avg":  round(statistics.fmean(s), 3),
            f"{self.name}_peak": round(max(s), 3),
            f"{self.name}_min":  round(min(s), 3),
            f"{self.name}_n":    len(s),
        }


# =====================================================================
# Goruntu on-isleme
# =====================================================================

def list_images(images_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    d = Path(images_dir)
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts])


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    import cv2
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=img.dtype)
    pad_h = (new_shape[0] - nh) // 2
    pad_w = (new_shape[1] - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas, r, (pad_w, pad_h)


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


# =====================================================================
# COCO mAP degerlendirme (runtime-bagimsiz)
# =====================================================================

def evaluate_map(predict_fn, gt_json: str, images_dir: str,
                 max_images=None, class_names=None, model_class_names=None):
    # predict_fn(image_bgr) -> (boxes_xyxy_pixel, scores, class_idx_0based)
    #   class_idx MODELIN cikti sirasinda. model_class_names verilirse isimle
    #   gt kategoriye eslenir; None ise POZISYON-tabanli (bu proje icin dogru).
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json))
    cat_ids = sorted(coco_gt.getCatIds())
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    if class_names is None:
        class_names = [cats[c] for c in cat_ids]

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    if model_class_names is not None:
        name2catid = {_norm(n): cid for cid, n in cats.items()}
        idx2catid = {}
        for i, nm in enumerate(model_class_names):
            cid = name2catid.get(_norm(nm))
            if cid is None:
                print(f"[mAP] uyari: model sinifi '{nm}' gt'de yok, atlanir")
            idx2catid[i] = cid
    else:
        idx2catid = {i: cat_ids[i] for i in range(len(cat_ids))}

    img_dir = Path(images_dir)
    img_infos = coco_gt.loadImgs(coco_gt.getImgIds())
    if max_images:
        img_infos = img_infos[:max_images]

    results, used_img_ids = [], []
    for info in img_infos:
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            continue
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        used_img_ids.append(info["id"])
        boxes, scores, labels = predict_fn(image_bgr)
        for box, score, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            cid = idx2catid.get(int(lab))
            if cid is None:
                continue
            results.append({
                "image_id":    int(info["id"]),
                "category_id": int(cid),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(score),
            })

    out = {"map50": 0.0, "map5095": 0.0, "per_class": {}, "n_images": len(used_img_ids)}
    if not results:
        print("[mAP] uyari: hic tahmin yok -> mAP=0")
        return out

    coco_dt = coco_gt.loadRes(results)
    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ce.params.imgIds = sorted(used_img_ids)
    ce.evaluate(); ce.accumulate(); ce.summarize()
    out["map5095"] = round(float(ce.stats[0]), 4)
    out["map50"]   = round(float(ce.stats[1]), 4)

    prec = ce.eval["precision"]
    for k, cid in enumerate(cat_ids):
        p50 = prec[0, :, k, 0, 2]
        p50 = p50[p50 > -1]
        ap50 = round(float(p50.mean()), 4) if p50.size else None
        out["per_class"][cats[cid]] = {"ap50": ap50}
    return out


def derive_efficiency(metrics: dict) -> dict:
    out = {}
    fps = metrics.get("fps_mean")
    pw = metrics.get("power_w_avg")
    lat = metrics.get("latency_mean_ms")
    if fps and pw:
        out["fps_per_watt"] = round(fps / pw, 3)
    if pw and lat:
        out["energy_mj_per_frame"] = round(pw * lat, 2)   # W * ms = mJ
    if metrics.get("map50") and pw:
        out["map50_per_watt"] = round(metrics["map50"] / pw, 4)
    params_m = metrics.get("params_m")
    if metrics.get("map50") and params_m:
        out["map50_per_mparam"] = round(metrics["map50"] / params_m, 4)
    return out


# =====================================================================
# SIRALAMA (ranking) — kullanicinin istedigi "sirala"
# =====================================================================

# Hangi metriklere gore siralanir; (anahtar, yon, etiket).
# yon: "desc" -> buyuk daha iyi, "asc" -> kucuk daha iyi.
RANK_METRICS = [
    ("fps_mean",            "desc", "FPS"),
    ("map50",               "desc", "mAP@0.5"),
    ("map5095",             "desc", "mAP@0.5:0.95"),
    ("latency_mean_ms",     "asc",  "latency"),
    ("fps_per_watt",        "desc", "FPS/W"),
    ("energy_mj_per_frame", "asc",  "enerji/frame"),
    ("disk_mb",             "asc",  "disk"),
]


def rank_models(rows: list, metric: str, ascending: bool) -> list:
    # Belirli bir metrige gore siralanmis (degeri olan) satirlari doner.
    have = [r for r in rows if r.get(metric) is not None]
    return sorted(have, key=lambda r: r[metric], reverse=not ascending)


def composite_score(rows: list) -> list:
    # Tum modeller arasinda min-max normalize edilmis bilesik skor (0..1):
    #   0.45*FPS + 0.45*mAP@0.5 + 0.10*(1/disk).  Daha buyuk = daha iyi denge.
    # Guc verisi varsa (cihaz kosumu) FPS/W de eklenir.
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    def norm(val, vals, invert=False):
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 0.5
        n = (val - lo) / (hi - lo)
        return 1.0 - n if invert else n

    fps_vals  = col("fps_mean")
    map_vals  = col("map50")
    disk_vals = col("disk_mb")
    fpw_vals  = col("fps_per_watt")
    has_power = len(fpw_vals) > 0

    for r in rows:
        parts, weights = [], []
        if r.get("fps_mean") is not None:
            parts.append(norm(r["fps_mean"], fps_vals)); weights.append(0.40)
        if r.get("map50") is not None:
            parts.append(norm(r["map50"], map_vals)); weights.append(0.40)
        if r.get("disk_mb") is not None:
            parts.append(norm(r["disk_mb"], disk_vals, invert=True)); weights.append(0.10)
        if has_power and r.get("fps_per_watt") is not None:
            parts.append(norm(r["fps_per_watt"], fpw_vals)); weights.append(0.10)
        if parts:
            wsum = sum(weights)
            r["score"] = round(sum(p * w for p, w in zip(parts, weights)) / wsum, 4)
        else:
            r["score"] = None
    return rank_models(rows, "score", ascending=False)


# =====================================================================
# Raporlama (JSON + CSV + leaderboard)
# =====================================================================

def save_results(results: dict, out_dir: str, tag: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"{tag}_{ts}.json"
    csv_path  = out / f"{tag}_{ts}.csv"
    rank_path = out / f"{tag}_{ts}_ranking.txt"

    rows = results.get("models", [])

    # Bilesik skor + her metrik icin siralama -> sonuca yaz.
    ranked = composite_score(rows)
    results["ranking_by_score"] = [r.get("name") for r in ranked]
    results["ranking"] = {}
    for metric, direction, _label in RANK_METRICS:
        order = rank_models(rows, metric, ascending=(direction == "asc"))
        results["ranking"][metric] = [
            {"name": r.get("name"), metric: r.get(metric)} for r in order
        ]

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV (skalar kolonlar).
    if rows:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols and not isinstance(r[k], (dict, list)):
                    cols.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    leaderboard = _format_leaderboard(results, tag)
    with open(rank_path, "w") as f:
        f.write(leaderboard)
    print("\n" + leaderboard)
    print(f"JSON: {json_path}\nCSV:  {csv_path}\nSIRA: {rank_path}")
    return {"json": str(json_path), "csv": str(csv_path), "ranking": str(rank_path)}


def _format_leaderboard(results: dict, tag: str) -> str:
    rows = results.get("models", [])
    lines = []
    lines.append("=" * 78)
    lines.append(f"EDGE PERF LEADERBOARD — {tag}")
    dev = results.get("device_info", {})
    if dev:
        lines.append(f"platform: {dev.get('platform', '?')}")
    lines.append("=" * 78)

    # Ana tablo: bilesik skora gore sirali.
    ranked = sorted([r for r in rows if r.get("score") is not None],
                    key=lambda r: r["score"], reverse=True)
    rest = [r for r in rows if r.get("score") is None]
    ordered = ranked + rest

    hdr = (f"{'#':>2}  {'model/variant':34s} {'score':>6} {'FPS':>7} "
           f"{'mAP@.5':>7} {'mAP.5:.95':>9} {'lat ms':>7} {'W':>6} "
           f"{'mJ/f':>7} {'disk MB':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(ordered, 1):
        def g(k, fmt="{}"):
            v = r.get(k)
            return fmt.format(v) if v is not None else "-"
        lines.append(
            f"{i:>2}  {str(r.get('name','?')):34s} "
            f"{g('score','{:.3f}'):>6} {g('fps_mean','{:.1f}'):>7} "
            f"{g('map50','{:.3f}'):>7} {g('map5095','{:.3f}'):>9} "
            f"{g('latency_mean_ms','{:.2f}'):>7} {g('power_w_avg','{:.2f}'):>6} "
            f"{g('energy_mj_per_frame','{:.1f}'):>7} {g('disk_mb','{:.1f}'):>8}"
        )

    # Tek-metrik sampiyonlari.
    lines.append("")
    lines.append("Metrik sampiyonlari:")
    for metric, direction, label in RANK_METRICS:
        order = rank_models(rows, metric, ascending=(direction == "asc"))
        if order:
            best = order[0]
            lines.append(f"  {label:14s}: {best.get('name'):34s} = {best.get(metric)}")
    lines.append("=" * 78)
    return "\n".join(lines)

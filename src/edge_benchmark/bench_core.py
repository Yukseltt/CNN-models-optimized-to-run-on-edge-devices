# bench_core.py
#
# Uc cihaz (edge) benchmark'lari icin ortak, cihaz-bagimsiz cekirdek.
# Shared device-agnostic core for edge benchmarks (Jetson Orin Nano, RPi5+Hailo).
#
# Icerik:
#   - LatencyStats: gecikme istatistikleri (mean/median/p90/p95/p99/std/min/max + FPS)
#   - BackgroundSampler: bir callable'i arka planda ornekleyip avg/peak/min doner
#                        (guc/sicaklik/util ornekleme icin; cihaza ozel callable verilir)
#   - evaluate_map: bir predict_fn'i test seti uzerinde kosturup pycocotools mAP hesaplar
#   - preprocess helpers: letterbox + resize
#   - save_results: JSON + CSV + konsol ozeti (mevcut bechmarks/*.json semasini genisletir)
#
# Cihaza ozel (guc okuma, runtime/engine/hef yukleme) kisimlar device script'lerinde.

import csv
import json
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
    # Olcum listesinden (ms) gecikme + FPS istatistikleri.
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
    # Verilen sample_fn()'i (float doner; None ise atlanir) interval saniyede bir
    # arka planda ornekler. start()/stop() arasinda toplanan degerlerin
    # avg/peak/min'ini summary() ile verir. Cihaza ozel guc/sicaklik okuyucu
    # bir lambda olarak verilir; okuma basarisizsa None dondurmesi yeterli.
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
    # YOLO tarzi en-boy koruyan resize + pad. (img: HxWx3 BGR/RGB ndarray)
    # Donus: padded_img, scale_ratio, (pad_w, pad_h) -> kutu geri-olcekleme icin.
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


# =====================================================================
# Canli kamera (webcam) FPS — uctan uca pipeline
# =====================================================================

def run_live_camera(infer_fn, source=0, warmup=15, measure=300, max_seconds=20,
                    frame_grabber=None, on_frame=None):
    # Canli akistan uctan uca FPS olcer: yakalama -> infer -> (post).
    # Statik latency'den FARKLI: kamera I/O + jitter dahil, gercek deployment hizi.
    #
    #   infer_fn(frame_bgr) -> (boxes, scores, labels)
    #   source: cv2.VideoCapture kaynagi (0 = ilk webcam, veya /dev/video0, dosya)
    #   frame_grabber: opsiyonel ozel kare kaynagi (test icin); None -> cv2 webcam
    #   measure/max_seconds: hangisi once dolarsa o.
    #
    # Donus:
    #   live_fps_e2e   : yakalama+infer dahil gercek FPS (kullanicinin gordugu)
    #   infer_only_fps : sadece model
    #   capture_fps    : kamera yakalama tavani
    #   camera_bound   : kamera mi darbogaz (infer_fps >> live_fps)
    #   e2e_lat_*      : uctan uca gecikme mean/p95/p99
    #   fps_first_half / fps_second_half / fps_drop_pct : sustained throttle gostergesi
    cap = None
    if frame_grabber is None:
        import cv2
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"kamera acilamadi: {source}")

        def grab():
            ok, f = cap.read()
            return f if ok else None
        frame_grabber = grab

    try:
        for _ in range(warmup):
            f = frame_grabber()
            if f is not None:
                infer_fn(f)

        cap_t, inf_t, e2e_t, stamps = [], [], [], []
        t_start = time.perf_counter()
        i = 0
        while i < measure:
            if max_seconds and (time.perf_counter() - t_start) > max_seconds:
                break
            t0 = time.perf_counter()
            f = frame_grabber()
            t1 = time.perf_counter()
            if f is None:
                break
            infer_fn(f)
            t2 = time.perf_counter()
            cap_t.append((t1 - t0) * 1000.0)
            inf_t.append((t2 - t1) * 1000.0)
            e2e_t.append((t2 - t0) * 1000.0)
            stamps.append(t2 - t_start)
            if on_frame is not None:
                on_frame(f)
            i += 1
    finally:
        if cap is not None:
            cap.release()

    out = {"live_frames": len(e2e_t), "live_duration_s": round(stamps[-1], 2) if stamps else 0.0}
    if not e2e_t:
        return out
    dur = stamps[-1] if stamps[-1] > 0 else 1e-9
    out["live_fps_e2e"]   = round(len(e2e_t) / dur, 2)
    out["infer_only_fps"] = round(1000.0 / statistics.fmean(inf_t), 2)
    out["capture_fps"]    = round(1000.0 / statistics.fmean(cap_t), 2) if statistics.fmean(cap_t) > 0 else None
    out["camera_bound"]   = bool(out["infer_only_fps"] > 1.25 * out["live_fps_e2e"])
    st = latency_stats(e2e_t)
    out["e2e_lat_mean_ms"] = st["latency_mean_ms"]
    out["e2e_lat_p95_ms"]  = st["latency_p95_ms"]
    out["e2e_lat_p99_ms"]  = st["latency_p99_ms"]
    # Sustained: ilk yari vs son yari FPS (throttle / isinma etkisi).
    half = len(stamps) // 2
    if half >= 2:
        t_mid = stamps[half - 1]
        fps1 = half / t_mid if t_mid > 0 else 0
        fps2 = (len(stamps) - half) / (stamps[-1] - t_mid) if (stamps[-1] - t_mid) > 0 else 0
        out["fps_first_half"]  = round(fps1, 2)
        out["fps_second_half"] = round(fps2, 2)
        out["fps_drop_pct"]    = round((1 - fps2 / fps1) * 100.0, 1) if fps1 > 0 else None
    return out


# =====================================================================
# COCO mAP degerlendirme (runtime-bagimsiz)
# =====================================================================

def evaluate_map(predict_fn, gt_json: str, images_dir: str,
                 max_images=None, class_names=None, model_class_names=None):
    # predict_fn(image_bgr_ndarray) -> (boxes_xyxy, scores, class_idx_0based)
    #   boxes ORIJINAL goruntu koordinatlarinda (xyxy, piksel).
    #   class_idx 0-based, MODELIN cikti sirasinda.
    #
    # model_class_names: modelin cikti sirasindaki sinif isimleri (orn YOLO
    #   ['car','person','other_vehicle']). Verilirse predicted idx -> ISIM ->
    #   gt category_id ile eslenir (sira/isimlendirme farkliligini cozer).
    #   None ise index dogrudan gt kategori sirasina denk varsayilir.
    #
    # gt_json: COCO _annotations.coco.json yolu.
    # Donus: {map50, map5095, per_class{...}, n_images}
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json))
    cat_ids = sorted(coco_gt.getCatIds())          # ornn [0,1,2]
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    if class_names is None:
        class_names = [cats[c] for c in cat_ids]

    # Model cikti index'i -> gt category_id eslemesi.
    # VARSAYILAN: pozisyon-tabanli (idx i -> sirali gt kategori i). Bu projede
    # tum modeller (frcnn/rtdetr Person/Car/OtherVehicle sirasinda; YOLO da gt
    # sirasinda tespit ediyor) icin dogru. DIKKAT: YOLO model.names bu projede
    # gt ile TUTARSIZ (model '0:car' der ama 0 aslinda person) -> isim-tabanli
    # esleme YANLIS olur; o yuzden model_class_names yalnizca isimleri gt ile
    # GERCEKTEN uyusan modeller icin acikca verilmeli.
    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    if model_class_names is not None:
        name2catid = {_norm(n): cid for cid, n in cats.items()}
        idx2catid = {}
        for i, nm in enumerate(model_class_names):
            cid = name2catid.get(_norm(nm))
            if cid is None:
                print(f"[mAP] uyari: model sinifi '{nm}' gt kategorilerinde yok, atlanir")
            idx2catid[i] = cid
    else:
        idx2catid = {i: cat_ids[i] for i in range(len(cat_ids))}

    img_dir = Path(images_dir)
    img_infos = coco_gt.loadImgs(coco_gt.getImgIds())
    if max_images:
        img_infos = img_infos[:max_images]

    results = []
    used_img_ids = []
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

    # Per-class AP@0.5
    prec = ce.eval["precision"]    # [T,R,K,A,M]
    for k, cid in enumerate(cat_ids):
        p50 = prec[0, :, k, 0, 2]
        p50 = p50[p50 > -1]
        ap50 = round(float(p50.mean()), 4) if p50.size else None
        out["per_class"][cats[cid]] = {"ap50": ap50}
    return out


# =====================================================================
# Raporlama
# =====================================================================

def save_results(results: dict, out_dir: str, tag: str) -> dict:
    # results: {device_info, models:[{name, ...metrics...}], ...}
    # JSON + CSV yazar, konsola ozet basar. Yazilan dosya yollarini doner.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"{tag}_{ts}.json"
    csv_path = out / f"{tag}_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = results.get("models", [])
    if rows:
        # Tum modellerdeki anahtarlarin birlesimi -> stabil kolon sirasi.
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

    print("\n" + "=" * 70)
    print(f"BENCHMARK OZETI ({tag})")
    print("=" * 70)
    for r in rows:
        print(f"  {r.get('name','?'):32s} "
              f"FPS={r.get('fps_mean','-')}  "
              f"lat={r.get('latency_mean_ms','-')}ms  "
              f"p99={r.get('latency_p99_ms','-')}ms  "
              f"mAP@.5={r.get('map50','-')}  "
              f"W={r.get('power_w_avg','-')}  "
              f"mJ/f={r.get('energy_mj_per_frame','-')}")
    print(f"\nJSON: {json_path}\nCSV:  {csv_path}")
    return {"json": str(json_path), "csv": str(csv_path)}


def derive_efficiency(metrics: dict) -> dict:
    # Turetilmis verimlilik metrikleri (varsa guc/mAP/param'dan).
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

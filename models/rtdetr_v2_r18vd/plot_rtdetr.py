"""RT-DETR egitim metriklerinden PNG grafikleri uretir.

Run klasoru: runs/<NAME>/training_metrics.xlsx
Cikti      : runs/<NAME>/plots/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


LW   = 1.8
DPI  = 150


def _to_float(values):
    out = []
    for v in values:
        if v is None or v == "":
            out.append(np.nan)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
    return np.array(out, dtype=np.float64)


def _load(xlsx_path: Path) -> dict:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Training Metrics"]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    data_rows = [r for r in rows[1:] if any(c is not None for c in r)]

    def col(name):
        idx = headers.index(name)
        return _to_float([r[idx] for r in data_rows])

    class_names = []
    for h in headers:
        if h.startswith("mAP (") and h.endswith(")"):
            class_names.append(h[len("mAP ("):-1])

    data = {
        "epoch":    col("Epoch"),
        "step":     col("Step"),
        "lr":       col("Learning Rate"),
        "t_loss":   col("Train Loss"),
        "v_loss":   col("Val Loss"),
        "map":      col("mAP@0.5:0.95"),
        "map50":    col("mAP@0.5"),
        "map75":    col("mAP@0.75"),
        "mar1":     col("mAR@1"),
        "mar10":    col("mAR@10"),
        "mar100":   col("mAR@100"),
        "class_names": class_names,
        "per_class": {
            n: {
                "map":    col(f"mAP ({n})"),
                "mar100": col(f"mAR@100 ({n})"),
            } for n in class_names
        },
    }
    return data


def _style(ax, title, xlabel="Epoch", ylabel=""):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def _save_loss(d, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d["epoch"], d["t_loss"], color="#2166AC", linewidth=LW, label="Train Loss")
    ax.plot(d["epoch"], d["v_loss"], color="#D73027", linewidth=LW,
            linestyle="--", label="Val Loss")
    _style(ax, "Training / Validation Loss", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=DPI)
    plt.close(fig)


def _save_map(d, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d["epoch"], d["map50"],  color="#2166AC", linewidth=LW, label="mAP@0.5")
    ax.plot(d["epoch"], d["map"],    color="#1B7837", linewidth=LW, label="mAP@0.5:0.95")
    ax.plot(d["epoch"], d["map75"],  color="#D73027", linewidth=LW,
            linestyle="--", label="mAP@0.75")
    _style(ax, "Overall mAP", ylabel="mAP")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "map_overall.png", dpi=DPI)
    plt.close(fig)


def _save_mar(d, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(d["epoch"], d["mar1"],   color="#5AAE61", linewidth=LW, label="mAR@1")
    ax.plot(d["epoch"], d["mar10"],  color="#1B7837", linewidth=LW, label="mAR@10")
    ax.plot(d["epoch"], d["mar100"], color="#00441B", linewidth=LW, label="mAR@100")
    _style(ax, "Mean Average Recall", ylabel="mAR")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / "mar_overall.png", dpi=DPI)
    plt.close(fig)


def _save_lr(d, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["epoch"], d["lr"], color="#762A83", linewidth=LW, label="LR")
    _style(ax, "Learning Rate Schedule", ylabel="LR")
    fig.tight_layout()
    fig.savefig(out / "lr.png", dpi=DPI)
    plt.close(fig)


def _save_per_class(d, out: Path, metric_key: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#2166AC", "#D73027", "#1B7837", "#762A83", "#B2182B", "#5AAE61"]
    for i, name in enumerate(d["class_names"]):
        ax.plot(
            d["epoch"], d["per_class"][name][metric_key],
            color=palette[i % len(palette)], linewidth=LW, label=name,
        )
    _style(ax, title, ylabel=title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=DPI)
    plt.close(fig)


def _save_dashboard(d, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(d["epoch"], d["t_loss"], label="Train", linewidth=LW)
    axes[0, 0].plot(d["epoch"], d["v_loss"], label="Val", linewidth=LW, linestyle="--")
    axes[0, 0].set_title("Loss"); axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(d["epoch"], d["map50"], label="mAP@0.5", linewidth=LW)
    axes[0, 1].plot(d["epoch"], d["map"],   label="mAP@0.5:0.95", linewidth=LW)
    axes[0, 1].plot(d["epoch"], d["map75"], label="mAP@0.75", linewidth=LW, linestyle="--")
    axes[0, 1].set_title("mAP"); axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(d["epoch"], d["mar1"],   label="mAR@1",   linewidth=LW)
    axes[1, 0].plot(d["epoch"], d["mar10"],  label="mAR@10",  linewidth=LW)
    axes[1, 0].plot(d["epoch"], d["mar100"], label="mAR@100", linewidth=LW)
    axes[1, 0].set_title("mAR"); axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()

    palette = ["#2166AC", "#D73027", "#1B7837", "#762A83"]
    for i, name in enumerate(d["class_names"]):
        axes[1, 1].plot(
            d["epoch"], d["per_class"][name]["map"],
            color=palette[i % len(palette)], linewidth=LW, label=name,
        )
    axes[1, 1].set_title("Per-class mAP@0.5:0.95"); axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out / "dashboard.png", dpi=DPI)
    plt.close(fig)


def make_plots(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    xlsx = run_dir / "training_metrics.xlsx"
    out = run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    d = _load(xlsx)
    if len(d["epoch"]) == 0:
        print(f"[plot_rtdetr] {xlsx} bos, grafik uretilmedi.")
        return

    _save_loss(d, out)
    _save_map(d, out)
    _save_mar(d, out)
    _save_lr(d, out)
    if d["class_names"]:
        _save_per_class(d, out, "map",    "Per-class mAP@0.5:0.95", "per_class_map.png")
        _save_per_class(d, out, "mar100", "Per-class mAR@100",      "per_class_mar100.png")
    _save_dashboard(d, out)
    print(f"[plot_rtdetr] {len(d['epoch'])} epoch icin grafikler -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Run klasoru (training_metrics.xlsx iceren)")
    args = p.parse_args()
    make_plots(args.run_dir)

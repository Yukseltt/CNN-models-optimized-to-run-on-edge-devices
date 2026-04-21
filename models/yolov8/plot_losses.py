"""
plot_losses.py
==============
YOLOv8 eğitim metriklerini results.csv'den okuyarak grafik üretir.

KULLANIM:
    python plot_losses.py --csv results.csv --out ./plots --skip 3

PARAMETRELER:
    --csv   : Ultralytics'in ürettiği results.csv dosyasının yolu
    --out   : Grafiklerin kaydedileceği klasör (varsayılan: ./plots)
    --skip  : Baştan atlanacak epoch sayısı (varsayılan: 3)
    --dpi   : Görüntü çözünürlüğü (varsayılan: 150)
    --fmt   : Dosya formatı — png, pdf, svg (varsayılan: png)

ÜRETİLEN GRAFİKLER:
    loss_box.{fmt}          — Box Loss (train + val)
    loss_cls.{fmt}          — Class Loss (train + val)
    loss_dfl.{fmt}          — DFL Loss (train + val)
    loss_three_combined.{fmt} — Box + Cls + DFL tek grafikte (train + val)
    loss_combined.{fmt}     — 6 panel: 3 loss + precision + recall + mAP
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────
# Stil sabitleri
# ──────────────────────────────────────────────────────────────

TRAIN_COLOR  = '#2166AC'   # koyu mavi
VAL_COLOR    = '#D73027'   # kırmızı
CLS_COLOR    = '#D73027'   # class loss için
DFL_COLOR    = '#1A9850'   # dfl loss için
LW           = 2.0

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.grid':         True,
    'grid.color':        '#e0e0e0',
    'grid.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '#cccccc',
})


# ──────────────────────────────────────────────────────────────
# Veri okuma
# ──────────────────────────────────────────────────────────────

def load_csv(path: str, skip: int) -> dict:
    """
    Ultralytics results.csv dosyasını okur.

    Beklenen sütunlar:
        epoch, time,
        train/box_loss, train/cls_loss, train/dfl_loss,
        metrics/precision(B), metrics/recall(B),
        metrics/mAP50(B), metrics/mAP50-95(B),
        val/box_loss, val/cls_loss, val/dfl_loss,
        lr/pg0, lr/pg1, lr/pg2
    """
    import pathlib
    if pathlib.Path(path).suffix.lower() in ('.xlsx', '.xls'):
        import pandas as pd
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        rows = [{k: str(v) for k, v in row.items()} for row in df.to_dict(orient='records')]
    else:
        with open(path, newline='') as f:
            raw = f.read()
        import io
        first_line = raw.split('\n')[0]
        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        rows = [{k.strip(): v.strip() for k, v in row.items() if k} for row in reader]

    def col(key):
        return np.array([float(r[key]) for r in rows])

    data = {
        'epoch':    col('epoch'),
        't_box':    col('train/box_loss'),
        't_cls':    col('train/cls_loss'),
        't_dfl':    col('train/dfl_loss'),
        'v_box':    col('val/box_loss'),
        'v_cls':    col('val/cls_loss'),
        'v_dfl':    col('val/dfl_loss'),
        'prec':     col('metrics/precision(B)'),
        'rec':      col('metrics/recall(B)'),
        'map50':    col('metrics/mAP50(B)'),
        'map5095':  col('metrics/mAP50-95(B)'),
    }

    # İlk `skip` epoch'u çıkar
    mask = data['epoch'] > skip
    return {k: v[mask] for k, v in data.items()}


# ──────────────────────────────────────────────────────────────
# Yardımcı çizim fonksiyonları
# ──────────────────────────────────────────────────────────────

def _style_ax(ax, ep, title, ylabel='Loss'):
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_xlim(ep[0], ep[-1])
    ax.legend()


def plot_single_loss(ax, ep, train, val, title,
                     train_color=TRAIN_COLOR, val_color=VAL_COLOR):
    ax.plot(ep, train, color=train_color, linewidth=LW, label='Train Loss')
    ax.plot(ep, val,   color=val_color,   linewidth=LW,
            linestyle='--', label='Validation Loss')
    _style_ax(ax, ep, title)


# ──────────────────────────────────────────────────────────────
# Grafik üreticiler
# ──────────────────────────────────────────────────────────────

def save_box_loss(d, out, dpi, fmt):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_single_loss(ax, d['epoch'], d['t_box'], d['v_box'], 'Box Loss')
    plt.tight_layout()
    path = out / f'loss_box.{fmt}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Kaydedildi: {path}')


def save_cls_loss(d, out, dpi, fmt):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_single_loss(ax, d['epoch'], d['t_cls'], d['v_cls'], 'Class Loss',
                     train_color=CLS_COLOR, val_color='#FC8D59')
    plt.tight_layout()
    path = out / f'loss_cls.{fmt}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Kaydedildi: {path}')


def save_dfl_loss(d, out, dpi, fmt):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_single_loss(ax, d['epoch'], d['t_dfl'], d['v_dfl'], 'DFL Loss',
                     train_color=DFL_COLOR, val_color='#74C476')
    plt.tight_layout()
    path = out / f'loss_dfl.{fmt}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Kaydedildi: {path}')


def save_three_combined(d, out, dpi, fmt):
    """
    Box, Cls ve DFL — train ve val — tek grafikte.
    Her loss kendi rengiyle; train düz, val kesikli çizgi.
    """
    ep = d['epoch']
    fig, ax = plt.subplots(figsize=(11, 6))

    # Box — mavi
    ax.plot(ep, d['t_box'], color='#2166AC', linewidth=LW, label='Box — Train')
    ax.plot(ep, d['v_box'], color='#2166AC', linewidth=LW,
            linestyle='--', label='Box — Val')

    # Cls — kırmızı
    ax.plot(ep, d['t_cls'], color='#D73027', linewidth=LW, label='Cls — Train')
    ax.plot(ep, d['v_cls'], color='#D73027', linewidth=LW,
            linestyle='--', label='Cls — Val')

    # DFL — yeşil
    ax.plot(ep, d['t_dfl'], color='#1A9850', linewidth=LW, label='DFL — Train')
    ax.plot(ep, d['v_dfl'], color='#1A9850', linewidth=LW,
            linestyle='--', label='DFL — Val')

    ax.set_title('Box + Class + DFL Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(ep[0], ep[-1])
    ax.legend(ncol=3, loc='upper right')

    plt.tight_layout()
    path = out / f'loss_three_combined.{fmt}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Kaydedildi: {path}')


def save_combined_dashboard(d, out, dpi, fmt):
    """
    6 panel: Box Loss | Class Loss | DFL Loss
             Precision | Recall    | mAP
    """
    ep = d['epoch']
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('YOLOv8n — Eğitim Metrikleri',
                 fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # Üst satır: losslar
    for col_idx, (key_t, key_v, title, tc, vc) in enumerate([
        ('t_box', 'v_box', 'Box Loss',   '#2166AC', '#D73027'),
        ('t_cls', 'v_cls', 'Class Loss', '#D73027', '#FC8D59'),
        ('t_dfl', 'v_dfl', 'DFL Loss',   '#1A9850', '#74C476'),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.plot(ep, d[key_t], color=tc, linewidth=LW, label='Train Loss')
        ax.plot(ep, d[key_v], color=vc, linewidth=LW,
                linestyle='--', label='Validation Loss')
        _style_ax(ax, ep, title)

    # Alt satır: precision
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ep, d['prec'], color='#2166AC', linewidth=LW,
             label='Validation Precision')
    _style_ax(ax4, ep, 'Precision', ylabel='Precision')

    # recall
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ep, d['rec'], color='#2166AC', linewidth=LW,
             label='Validation Recall')
    _style_ax(ax5, ep, 'Recall', ylabel='Recall')

    # mAP
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ep, d['map50'],   color='#2166AC', linewidth=LW,
             label='mAP@0.5')
    ax6.plot(ep, d['map5095'], color='#D73027', linewidth=LW,
             linestyle='--', label='mAP@0.5:0.95')
    _style_ax(ax6, ep, 'mAP', ylabel='mAP')

    path = out / f'loss_combined.{fmt}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Kaydedildi: {path}')


# ──────────────────────────────────────────────────────────────
# Ana fonksiyon
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 results.csv → grafik üretici'
    )
    parser.add_argument('--csv',  default='results.csv',
                        help='results.csv yolu')
    parser.add_argument('--out',  default='./plots',
                        help='Çıktı klasörü')
    parser.add_argument('--skip', type=int, default=3,
                        help='Baştan atlanacak epoch sayısı')
    parser.add_argument('--dpi',  type=int, default=150)
    parser.add_argument('--fmt',  default='png',
                        choices=['png', 'pdf', 'svg'])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'CSV okunuyor: {args.csv}')
    d = load_csv(args.csv, skip=args.skip)
    print(f'  Toplam epoch: {len(d["epoch"])}  '
          f'(ilk {args.skip} epoch atlandı)')
    print()

    print('Grafikler üretiliyor...')
    save_box_loss(d, out, args.dpi, args.fmt)
    save_cls_loss(d, out, args.dpi, args.fmt)
    save_dfl_loss(d, out, args.dpi, args.fmt)
    save_three_combined(d, out, args.dpi, args.fmt)
    save_combined_dashboard(d, out, args.dpi, args.fmt)

    print()
    print(f'Tamamlandı. {args.out} klasörüne 5 grafik kaydedildi.')


if __name__ == '__main__':
    main()
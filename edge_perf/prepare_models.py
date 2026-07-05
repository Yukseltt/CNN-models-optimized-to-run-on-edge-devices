#!/usr/bin/env python3
# prepare_models.py
#
# 5 modelin base (duz) + pruned .pt agirliklarini edge_perf/models/<key>/ altina
# yerlestirir:  models/<key>/base.pt , models/<key>/pruned.pt
#
# Sonra convert/export_onnx.py + convert/quantize_onnx.py ayni klasore .onnx,
# _fp16.onnx, _int8.onnx ekler. Boylece models/ TEK-PAKET tasinabilir olur
# (Raspberry / Jetson'a kopyala -> calistir).
#
# Varsayilan: KOPYA (cihaza tasinabilsin). Sadece bu sunucuda denemek icin
# --link ile symlink (disk sismez) kullanabilirsin.
#
#   python edge_perf/prepare_models.py            # gercek kopya (deploy-hazir)
#   python edge_perf/prepare_models.py --link     # symlink (yalniz bu sunucu)

import argparse
import shutil

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true",
                    help="kopya yerine symlink (yalniz bu sunucuda gecerli)")
    args = ap.parse_args()

    print(f"hedef: {config.MODELS_DIR}  (mod: {'symlink' if args.link else 'kopya'})")
    n_ok, n_miss = 0, 0
    for model in config.MODELS:
        d = config.model_dir(model["key"])
        d.mkdir(parents=True, exist_ok=True)
        for structure in ("normal", "pruned"):
            cands = model["normal_pt"] if structure == "normal" else model["pruned_pt"]
            src = config.first_existing(cands)
            dst = config.canonical_weight(model["key"], structure, "pt")
            tag = f"{model['key']}/{config.STRUCT_LABEL[structure]}.pt"
            if src is None:
                print(f"  [YOK] {tag:42s} -> aday yollarin hicbiri yok")
                n_miss += 1
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.link:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            print(f"  [OK]  {tag:42s} <- {src.relative_to(config.PROJECT_ROOT)} "
                  f"({src.stat().st_size / 1e6:.1f} MB)")
            n_ok += 1

    print(f"\nbitti: {n_ok} yerlesti, {n_miss} eksik.")
    print("sonraki adim: python edge_perf/convert/export_onnx.py && "
          "python edge_perf/convert/quantize_onnx.py")


if __name__ == "__main__":
    main()

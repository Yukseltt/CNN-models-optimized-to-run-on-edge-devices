# prune_common.py
#
# 5 model icin ortak pruning yardimcilari.
# Shared helpers for the 5-model pruning pipelines.
#
# Icerik:
#   - set_mp_sharing(): bu host'ta DataLoader FD tukenmesini onleyen sharing stratejisi.
#   - count_params(): toplam parametre sayaci.
#   - reduction_pct(): kesim sonrasi % azalma.
#   - slim_onnx(): ONNX modelini onnxsim ile sadelestir (CPU affinity patch'i ile).
#
# Pruning mantigi ayri modullerde (her mimari ailesi farkli yol istiyor):
#   - prune_lib_yolo.py    : Ultralytics YOLO (tam structured + sensitivity sweep)
#   - prune_lib_frcnn.py   : torchvision Faster R-CNN (backbone-only DepGraph)
#   - prune_lib_rtdetr.py  : HF RT-DETR v2 (backbone-only DepGraph)

import os

import torch.multiprocessing as mp


def set_mp_sharing():
    # Bu host'ta DataLoader worker'lari + paylasimli bellek FD'leri tuketip EMFILE
    # (too many open files) veriyor; fine-tune egitiminden once "file_system"
    # stratejisi sart (yoksa egitim takilir/coker).
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def count_params(model):
    # Modelin toplam parametre sayisi (kesim oncesi/sonrasi gercek boyut icin).
    return sum(p.numel() for p in model.parameters())


def reduction_pct(before, after):
    # Kesim sonrasi yuzde azalma.
    if before <= 0:
        return 0.0
    return round((1.0 - after / before) * 100.0, 2)


def build_budget(sensitivity, prunable_names, min_floor, max_ceil, step=0.05):
    # Sensitivity sweep sonuclarindan per-layer kesim orani bütçesi uretir.
    # Az hassas katman (dusuk ortalama mAP dususu) -> MAX_CEIL'e yakin,
    # cok hassas katman -> MIN_FLOOR'a yakin (MIN_FLOOR..MAX_CEIL araliginda).
    #
    #   sensitivity: {layer_name: {ratio: mAP_dususu}}
    #   prunable_names: butce uretilecek tum prunable katman isimleri
    #
    # Donus: (butce {name: oran}, skor {name: ortalama_dusus})
    skor = {n: sum(v.values()) / len(v) for n, v in sensitivity.items() if v}
    s_min = min(skor.values()) if skor else 0.0
    s_max = max(skor.values()) if skor else 0.0

    butce = {}
    for name in prunable_names:
        if name in skor and s_max > s_min:
            norm = (skor[name] - s_min) / (s_max - s_min)          # 0..1
            oran = max_ceil - norm * (max_ceil - min_floor)
        else:
            oran = min_floor
        butce[name] = float(round(round(oran / step) * step, 2))
    return butce, skor


def slim_onnx(onnx_path):
    # onnxsim ile ONNX modelini sadelestir.
    #
    # onnxsim icindeki onnxruntime, intra_op_num_threads belirtilmediginde worker
    # thread'leri sirali CPU indekslerine (0..N-1) pinlemeye calisir. Bu host'ta
    # izinli cekirdekler non-contiguous ({4,6,21,...}) oldugundan
    # "pthread_setaffinity_np failed ... error code: 22" hatasi basar.
    # Cozum (ORT mesajinin kendi onerisi): thread sayisini ACIKCA ver -> ORT
    # affinity pinlemeyi tamamen atlar. onnxsim kendi SessionOptions'ini
    # (intra_op=0/auto) uretir; o yuzden InferenceSession'i kisa sureligine
    # sarmalayip degeri set ediyoruz.
    try:
        import onnx
        import onnxruntime as ort
        from onnxsim import simplify
    except Exception as e:
        print("onnxsim/onnx import edilemedi, slim atlandi:", e)
        return

    n_threads = max(1, len(os.sched_getaffinity(0)))
    orig_session = ort.InferenceSession

    def _session_with_threads(*args, **kwargs):
        so = kwargs.get("sess_options")
        if so is None and len(args) >= 2 and isinstance(args[1], ort.SessionOptions):
            so = args[1]
        if so is not None and so.intra_op_num_threads == 0:
            so.intra_op_num_threads = n_threads
        return orig_session(*args, **kwargs)

    ort.InferenceSession = _session_with_threads
    try:
        m = onnx.load(str(onnx_path))
        m_simp, ok = simplify(m)
        if ok:
            onnx.save(m_simp, str(onnx_path))
            print("onnxsim: ok")
        else:
            print("onnxsim: basarisiz (check failed), orijinal korundu")
    except Exception as e:
        print("onnxsim atlandi:", e)
    finally:
        ort.InferenceSession = orig_session  # patch'i geri al

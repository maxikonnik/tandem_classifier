"""Frozen-backbone probe for the visual boundaries (drogue throw, canopy deploy).

DINOv2-small embeddings + a tiny linear probe (weights in ``deploy_probe.json``)
label each 1 fps frame in [exit, break-off] with its phase; the drogue and deploy
times are the first frames entering свободное падение and раскрытие. Validated
leave-one-session-out on 143 labelled jumps (97 Samples + 46 D:, 133 sessions):
deploy 88 %, drogue 90 %, median ~0.6 s, only 2 % physically-impossible collapses —
far above the hand-tuned visual heuristics (59 % / 16 %).

torch/transformers are OPTIONAL runtime deps, imported lazily. If they (or the
weights file) are missing, ``predict_boundaries`` returns None and the caller falls
back to the heuristic detectors.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile

import numpy as np

_PROBE_JSON = os.path.join(os.path.dirname(__file__), "deploy_probe.json")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_cfg = None
_model = None


def _load():
    global _cfg, _model
    if _cfg is None:
        with open(_PROBE_JSON, encoding="utf-8") as f:
            _cfg = json.load(f)
    if _model is None:
        from transformers import AutoModel  # lazy, optional dep
        _model = AutoModel.from_pretrained(_cfg["backbone"]).eval()
    return _cfg, _model


def _embed(frame_paths, model):
    import torch
    from PIL import Image
    out = []
    for fp in frame_paths:
        a = np.asarray(Image.open(fp).convert("RGB").resize((224, 224)), np.float32) / 255.0
        a = (a - _IMAGENET_MEAN) / _IMAGENET_STD
        with torch.no_grad():
            o = model(pixel_values=torch.tensor(a.transpose(2, 0, 1)[None]))
        out.append(o.pooler_output[0].numpy())
    return np.asarray(out, np.float32)


def _first_ge(cls, ts, k):
    for i in range(len(cls) - 1):
        if cls[i] >= k and cls[i + 1] >= k:
            return float(ts[i])
    for i in range(len(cls)):
        if cls[i] >= k:
            return float(ts[i])
    return None


def predict_boundaries(path: str, exit_t: float, breakoff_t: float):
    """Return ``{"drogue": t|None, "deploy": t|None}``, or None when the probe or its
    optional deps are unavailable / no frames could be read."""
    if not os.path.exists(_PROBE_JSON):
        return None
    try:
        cfg, model = _load()
    except Exception:
        return None

    lo, hi = max(0.0, exit_t - 1.0), breakoff_t + 1.0
    td = tempfile.mkdtemp()
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", f"{lo:.2f}", "-to", f"{hi:.2f}", "-i", path,
                        "-vf", "fps=1,scale=224:224", os.path.join(td, "f_%04d.jpg")],
                       check=False, capture_output=True)
        frames = sorted(glob.glob(os.path.join(td, "f_*.jpg")))
        if not frames:
            return None
        try:
            emb = _embed(frames, model)
        except Exception:
            return None
        ts = np.array([lo + i for i in range(len(frames))], np.float32)
    finally:
        for fp in glob.glob(os.path.join(td, "f_*.jpg")):
            os.remove(fp)
        os.rmdir(td)

    mu = np.array(cfg["mu"], np.float32); sd = np.array(cfg["sd"], np.float32)
    W = np.array(cfg["W"], np.float32); b = np.array(cfg["b"], np.float32)
    cls = (((emb - mu) / sd) @ W.T + b).argmax(1)
    sm = cls.copy()
    for i in range(len(cls)):
        sm[i] = np.bincount(cls[max(0, i - 1):i + 2], minlength=4).argmax()
    # phases: 0 отделение, 1 свободное падение, 2 раскрытие, 3 после
    return {"drogue": _first_ge(sm, ts, 1), "deploy": _first_ge(sm, ts, 2)}

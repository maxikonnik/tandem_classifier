"""Frozen-backbone probe for the visual phase boundaries (exit, drogue, deploy).

DINOv2-small embeddings + a tiny linear probe (weights in ``deploy_probe.json``)
label each frame with its phase; a boundary is the first sustained frame of the next
phase. The coarse pass runs at 1 fps; exit is then refined at ``REFINE_FPS`` inside a
±``REFINE_HALF_S`` window around the coarse transition, cutting the 1 s sampling step.

Phases are time-ordered. A 5-class probe starts with "до отделения" (the cabin) and
therefore finds exit on its own; the older 4-class probe starts at "отделение" and
gives drogue/deploy only. Both weight layouts are supported. Validated
leave-one-session-out on 153 labelled jumps (5-class, 140 sessions), share within 2 s:
drogue 92 %, deploy 92 % (4-class was 90 / 88 %), exit 98 %. Exit refined at 5 fps and
lag-calibrated: 98 % within 1 s, median 0.2 s — also with a grid not aligned to the
exit (DJI / no telemetry), where plain 1 fps gives 66 % within 1 s, median 0.8 s.

torch/transformers are OPTIONAL runtime deps, imported lazily. If they (or the
weights file) are missing, the predictors return None and the caller falls back to
the heuristic detectors.
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
_BATCH = 16

# Exit refinement: re-classify at 5 fps within ±3 s of the 1 fps transition and take
# the first run of REFINE_RUN consecutive frames past the cabin (0.6 s at 5 fps), so a
# single ambiguous door-frame does not trigger it.
REFINE_FPS = 5
REFINE_HALF_S = 3.0
REFINE_RUN = 3
REFINE_SMOOTH = 3
# The refined visual exit lands systematically late of the labelled exit (the
# physical push-off): the camera still shows the door for a few hundred ms after it,
# plus ffmpeg's fps filter hands slot n the last source frame before (n+0.5)/fps.
# Calibrated leave-one-session-out on 153 jumps (median signed error, other sessions
# only): +0.40 s. It holds only for REFINE_FPS=5 — re-calibrate if that changes.
EXIT_VISUAL_LAG_S = 0.40

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
    for s in range(0, len(frame_paths), _BATCH):
        batch = []
        for fp in frame_paths[s:s + _BATCH]:
            a = np.asarray(Image.open(fp).convert("RGB").resize((224, 224)), np.float32) / 255.0
            batch.append(((a - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))
        with torch.no_grad():
            out.append(model(pixel_values=torch.tensor(np.stack(batch))).pooler_output.numpy())
    return np.concatenate(out).astype(np.float32)


def _first_ge(cls, ts, k):
    for i in range(len(cls) - 1):
        if cls[i] >= k and cls[i + 1] >= k:
            return float(ts[i])
    for i in range(len(cls)):
        if cls[i] >= k:
            return float(ts[i])
    return None


def _boundary(sm, ts, k):
    """First sustained frame at or past phase ``k`` — but only if the span opens
    before it. A span that starts already past ``k`` never saw the transition (e.g. a
    DJI clip that begins under canopy), so the boundary is unknown, not frame 0."""
    if len(sm) == 0 or sm[0] >= k:
        return None
    return _first_ge(sm, ts, k)


def _first_run_ge(cls, ts, k, run):
    for i in range(len(cls) - run + 1):
        if all(cls[i + q] >= k for q in range(run)):
            return float(ts[i])
    return None


def _smooth(cls, n_classes, width=3):
    h = width // 2
    sm = cls.copy()
    for i in range(len(cls)):
        sm[i] = np.bincount(cls[max(0, i - h):i + h + 1], minlength=n_classes).argmax()
    return sm


def _classify_span(path, lo, hi, fps, cfg, model):
    """``(ts, class per frame)`` for frames sampled at ``fps`` in [lo, hi], or None."""
    td = tempfile.mkdtemp()
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", f"{lo:.2f}", "-to", f"{hi:.2f}", "-i", path,
                        "-vf", f"fps={fps},scale=224:224", os.path.join(td, "f_%04d.jpg")],
                       check=False, capture_output=True)
        frames = sorted(glob.glob(os.path.join(td, "f_*.jpg")))
        if not frames:
            return None
        try:
            emb = _embed(frames, model)
        except Exception:
            return None
    finally:
        for fp in glob.glob(os.path.join(td, "f_*.jpg")):
            os.remove(fp)
        os.rmdir(td)
    ts = np.array([lo + i / fps for i in range(len(emb))], np.float32)
    mu = np.array(cfg["mu"], np.float32); sd = np.array(cfg["sd"], np.float32)
    W = np.array(cfg["W"], np.float32); b = np.array(cfg["b"], np.float32)
    return ts, (((emb - mu) / sd) @ W.T + b).argmax(1)


def _refine(path, coarse_t, k, cfg, model, lag_s=0.0):
    """Re-locate a 1 fps boundary at REFINE_FPS within ±REFINE_HALF_S: the first run
    of REFINE_RUN consecutive frames at or past phase ``k``, minus the calibrated
    ``lag_s``. Falls back to the (uncalibrated) coarse time when the fine pass finds
    nothing — the coarse 1 fps estimate carries no such lag on average."""
    got = _classify_span(path, max(0.0, coarse_t - REFINE_HALF_S), coarse_t + REFINE_HALF_S,
                         REFINE_FPS, cfg, model)
    if got is None:
        return coarse_t
    ts, cls = got
    sm = _smooth(cls, len(cfg["phases"]), REFINE_SMOOTH)
    if sm[0] >= k:            # fine window opens past the transition: cannot place it
        return coarse_t
    fine = _first_run_ge(sm, ts, k, REFINE_RUN)
    return max(0.0, fine - lag_s) if fine is not None else coarse_t


def predict_boundaries(path: str, exit_t: float, breakoff_t: float):
    """Return ``{"exit": t|None, "drogue": t|None, "deploy": t|None}`` for the window
    around [exit, break-off], or None when the probe or its optional deps are
    unavailable / no frames could be read. ``exit`` is only produced by a probe that
    was trained with the pre-exit ("до отделения") class; the window then opens
    ``pre_s`` seconds before the telemetry exit so the probe sees the cabin."""
    if not os.path.exists(_PROBE_JSON):
        return None
    try:
        cfg, _ = _load()
    except Exception:
        return None
    pre = float(cfg.get("pre_s", 1.0))
    return predict_span(path, max(0.0, exit_t - pre), breakoff_t + 1.0)


def predict_span(path: str, lo: float, hi: float):
    """Classify the frames in [lo, hi] and return the phase boundaries found in it
    (see ``predict_boundaries``). Usable without telemetry: pass the whole clip and a
    pre-exit-aware probe locates exit, drogue and deploy on its own."""
    if not os.path.exists(_PROBE_JSON):
        return None
    try:
        cfg, model = _load()
    except Exception:
        return None
    got = _classify_span(path, lo, hi, 1, cfg, model)
    if got is None:
        return None
    ts, cls = got
    phases = cfg["phases"]
    sm = _smooth(cls, len(phases))
    exit_t = None
    if "до отделения" in phases:
        k = phases.index("отделение")
        exit_t = _boundary(sm, ts, k)
        if exit_t is not None:
            exit_t = _refine(path, exit_t, k, cfg, model, lag_s=EXIT_VISUAL_LAG_S)
    return {"exit": exit_t,
            "drogue": _boundary(sm, ts, phases.index("свободное падение")),
            "deploy": _boundary(sm, ts, phases.index("раскрытие"))}

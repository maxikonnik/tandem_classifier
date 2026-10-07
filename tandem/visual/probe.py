"""Frozen-backbone probe for the visual phase boundaries (exit, drogue, deploy).

DINOv2-small embeddings + a tiny linear probe (weights in ``deploy_probe.json``) give
per-frame phase log-probabilities. Phases are strictly ordered and each occupies one
contiguous run, so the boundaries are decoded jointly (segmental / HSMM decoding):
the split that maximises the frame log-probs plus a log-prior on how long each phase
lasts (fitted on the labelled jumps). That rules out, e.g., a "раскрытие" frame deep
in free-fall pulling deploy 40 s early. The coarse pass runs at 1 fps; exit is then
refined at ``REFINE_FPS`` inside ±``REFINE_HALF_S`` around it.

A 5-class probe starts with "до отделения" (the cabin) and therefore finds exit on its
own; the older 4-class probe starts at "отделение" and gives drogue/deploy only. Weights
without phase durations fall back to argmax + majority smoothing + first crossing.

Trained with post-break-off frames too, so break-off is found visually (the pair
leaving the frame) when the weights say ``visual_breakoff``. Validated
leave-one-session-out on 153 labelled jumps (140 sessions), share within 2 s with HSMM
decoding: exit 99 %, drogue 96 %, deploy 98 %, break-off 93 % (argmax: 98 / 90 / 91 /
91 %; the gyroscope break-off: 83 %, and none at all on 16 % of jumps). Exit refined at
5 fps and lag-calibrated: 98 % within 1 s, median 0.2 s — also with a sampling grid not
aligned to the exit (DJI / no telemetry), where plain 1 fps gives 66 %, median 0.8 s.

torch/transformers are OPTIONAL runtime deps, imported lazily. If they (or the weights
file) are missing, the predictors return None and the caller falls back to the
heuristic detectors.
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

# HSMM decoding: logits are divided by HSMM_TEMP before the log-softmax (the linear
# probe is over-confident; 1, 2 and 4 all beat argmax, 2 is the middle choice). Phase
# durations are log-normal mixed with a DUR_EPS uniform floor over [0, DUR_UMAX] s.
HSMM_TEMP = 2.0
DUR_EPS = 0.02
DUR_UMAX = 200.0

# Exit refinement: re-classify at 5 fps within ±3 s of the coarse transition and take
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


def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _duration_prior(params):
    """log-prior of a phase lasting x seconds: log-normal (mean/sd of log-seconds)
    mixed with a small uniform floor so outliers stay possible; flat without params."""
    if params is None:
        return lambda x: np.zeros_like(np.asarray(x, float))
    mu, sd = params

    def f(x):
        x = np.asarray(x, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.log(np.maximum(x, 1e-9))
            ln = np.where(x > 0, -lx - np.log(sd * np.sqrt(2 * np.pi))
                          - (lx - mu) ** 2 / (2 * sd * sd), -np.inf)
        return np.logaddexp(np.log(1 - DUR_EPS) + ln, np.log(DUR_EPS / DUR_UMAX))
    return f


def _decode(logp, priors, dt):
    """Split N frames into the ordered phases — each one contiguous run, possibly
    empty — maximising summed log-probs plus the duration log-prior of every phase
    after the first. Exact O(C·N²) dynamic programme; returns each phase's start index
    (N for a phase that never begins inside the span)."""
    N, C = logp.shape
    S = np.vstack([np.zeros(C), np.cumsum(logp, 0)])
    dp = np.full((C, N + 1), -np.inf)
    arg = np.zeros((C, N + 1), int)
    dp[0] = S[:, 0]
    for c in range(1, C):
        for b in range(N + 1):
            a = np.arange(b + 1)
            sc = dp[c - 1, a] + (S[b, c] - S[a, c]) + priors[c]((b - a) * dt)
            k = int(np.argmax(sc))
            dp[c, b] = sc[k]
            arg[c, b] = a[k]
    starts = [0] * C
    b = N
    for c in range(C - 1, 0, -1):
        starts[c] = int(arg[c, b])
        b = starts[c]
    return starts


def _embed_span(path, lo, hi, fps, model):
    """``(ts, backbone embedding per frame)`` for frames sampled at ``fps`` in [lo, hi],
    or None. Shared by the phase probe and the shot-scale head."""
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
    return np.array([lo + i / fps for i in range(len(emb))], np.float32), emb


def _classify_span(path, lo, hi, fps, cfg, model):
    """``(ts, logits per frame)`` for frames sampled at ``fps`` in [lo, hi], or None."""
    got = _embed_span(path, lo, hi, fps, model)
    if got is None:
        return None
    ts, emb = got
    mu = np.array(cfg["mu"], np.float32); sd = np.array(cfg["sd"], np.float32)
    W = np.array(cfg["W"], np.float32); b = np.array(cfg["b"], np.float32)
    return ts, ((emb - mu) / sd) @ W.T + b


def _refine(path, coarse_t, k, cfg, model, lag_s=0.0):
    """Re-locate a 1 fps boundary at REFINE_FPS within ±REFINE_HALF_S: the first run
    of REFINE_RUN consecutive frames at or past phase ``k``, minus the calibrated
    ``lag_s``. Falls back to the (uncalibrated) coarse time when the fine pass finds
    nothing — the coarse 1 fps estimate carries no such lag on average."""
    got = _classify_span(path, max(0.0, coarse_t - REFINE_HALF_S), coarse_t + REFINE_HALF_S,
                         REFINE_FPS, cfg, model)
    if got is None:
        return coarse_t
    ts, z = got
    sm = _smooth(z.argmax(1), len(cfg["phases"]), REFINE_SMOOTH)
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
    post = float(cfg.get("post_s", 1.0))
    return predict_span(path, max(0.0, exit_t - pre), breakoff_t + post)


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
    ts, z = got
    phases = cfg["phases"]
    if cfg.get("durations"):
        priors = [_duration_prior(cfg["durations"].get(p)) for p in phases]
        starts = _decode(_log_softmax(z / HSMM_TEMP), priors, dt=1.0)
        jump_seen = starts[1] < len(ts)          # some frame decoded past the first phase

        def at(name):
            s = starts[phases.index(name)]
            return float(ts[s]) if 0 < s < len(ts) else None   # outside the span = unseen
    else:
        sm = _smooth(z.argmax(1), len(phases))
        jump_seen = bool((sm >= 1).any())

        def at(name):
            return _boundary(sm, ts, phases.index(name))
    exit_t = None
    if "до отделения" in phases:
        exit_t = at("отделение")
        if exit_t is not None:
            exit_t = _refine(path, exit_t, phases.index("отделение"), cfg, model,
                             lag_s=EXIT_VISUAL_LAG_S)
    # jump_seen tells "only cabin in the span" (not a jump) apart from "the span opened
    # after the exit" (exit unseen, but the jump is there) — both give exit=None.
    out = {"exit": exit_t, "drogue": at("свободное падение"), "deploy": at("раскрытие"),
           "jump_seen": jump_seen if "до отделения" in phases else None}
    if cfg.get("visual_breakoff") and "после" in phases:
        out["breakoff"] = at("после")
    return out

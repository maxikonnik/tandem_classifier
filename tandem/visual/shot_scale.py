"""Shot scale: how much of the frame the tandem pair fills, from the probe's backbone.

A ridge head on the frozen DINOv2 embedding (the same 1 fps frames the phase probe
uses) predicts sqrt(pair share of the frame). It was distilled from an RT-DETR person
detector — the pair is the largest group of overlapping person boxes — so it costs
nothing beyond the embedding the probe computes anyway. Weights: shot_scale.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

_JSON = os.path.join(os.path.dirname(__file__), "shot_scale.json")
_head = None


def _load_head():
    global _head
    if _head is None:
        with open(_JSON, encoding="utf-8") as f:
            h = json.load(f)
        _head = (np.asarray(h["mu"], np.float32), np.asarray(h["sd"], np.float32),
                 np.asarray(h["w"], np.float32), float(h["b"]))
    return _head


def fraction_from_embeddings(emb) -> np.ndarray:
    """Pair share of the frame (0..1) for each backbone embedding row."""
    mu, sd, w, b = _load_head()
    return np.clip(((np.asarray(emb, np.float32) - mu) / sd) @ w + b, 0.0, 1.0) ** 2


def pair_fractions(path: str, lo: float, hi: float, fps: float = 1.0):
    """``(ts, pair share of frame)`` sampled at ``fps`` in [lo, hi], or None when the
    head, the backbone or the video is unavailable."""
    if not os.path.exists(_JSON):
        return None
    try:
        from tandem.visual import probe
        _, model = probe._load()
        got = probe._embed_span(path, lo, hi, fps, model)
    except Exception:
        return None
    if got is None:
        return None
    ts, emb = got
    return ts, fraction_from_embeddings(emb)

"""Reduce one grayscale keyframe to scalar visual features.

blob = the dark, contrasting object (the tandem pair against bright sky).
structure = fraction of high-gradient pixels (the close-up aircraft is
highly structured; open sky is smooth). No object detector, no ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DARK_REL = 0.5        # contrasting-dark: below DARK_REL * frame mean luma
GRAD_THRESH = 25.0    # a pixel is an "edge" if its gradient magnitude exceeds this
CENTER_FRAC = 0.4     # central box side, as a fraction of frame


@dataclass
class FrameFeature:
    t_s: float
    mean_luma: float
    blob_area_frac: float
    center_fill_frac: float
    structure_frac: float


def frame_features(gray, t_s: float = 0.0) -> FrameFeature:
    g = np.asarray(gray, dtype=np.float32)
    mean_luma = float(g.mean())
    dark = g < (mean_luma * DARK_REL)
    blob_area_frac = float(dark.mean())

    h, w = g.shape
    y0, y1 = int(h * (0.5 - CENTER_FRAC / 2)), int(h * (0.5 + CENTER_FRAC / 2))
    x0, x1 = int(w * (0.5 - CENTER_FRAC / 2)), int(w * (0.5 + CENTER_FRAC / 2))
    center = dark[y0:y1, x0:x1]
    center_fill_frac = float(center.mean()) if center.size else 0.0

    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    edge_frac = 0.5 * (float((gx > GRAD_THRESH).mean()) + float((gy > GRAD_THRESH).mean()))
    return FrameFeature(t_s=t_s, mean_luma=mean_luma, blob_area_frac=blob_area_frac,
                        center_fill_frac=center_fill_frac, structure_frac=edge_frac)

"""Visual canopy-deploy detector — the last burst of bright suspension lines.

Signature of the MAIN deploy, from the operator's camera: as the canopy pulls out
of the d-bag, its BRIGHT suspension lines (светлые стропы) snap taut and fan out —
many thin, bright, near-vertical lines against the sky. Ground-truth frames show
this is the reliable cue: at the deploy the line count spikes (up to ~10-18), while
the object itself is bright (not a dark blob). A dark-blob cue was tried and
rejected: at the real deploy the dark fraction is often ~0, and the only strong
dark blob nearby is the free-fall HELMET of the pair — so a dark-blob detector
anchors on the helmet, not the deploy.

Two events produce line bursts in the window: the DROGUE (earlier, throughout
free-fall) and the MAIN deploy (later, just before the operator turns away). They
are separated in TIME: the main deploy is the LAST line burst before break-off. So
the anchor is the peak of the last bright-line burst in the pre-break-off window.
The free-fall helmet carries few lines (< the burst threshold) and is ignored; the
operator's own later deployment is excluded by ending the window before break-off.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

# band where the suspension lines live (from the canopy near the top down to the pair)
CX0, CX1, CY0, CY1 = 0.25, 0.75, 0.03, 0.78
RIDGE_D = 2                 # neighbour distance for the thin-line test (px, at ANALYZE_HEIGHT)
RIDGE_T = 16.0             # a line pixel is this much brighter than both neighbours
COL_FRAC = 0.12            # a "line column" has ridge pixels over at least this fraction of the band
LINE_BURST_MIN = 3.0      # a deploy line burst reaches at least this many line columns; a drogue's
                          # single line and a free-fall helmet stay below it
BURST_LOOKBACK_S = 1.75   # once the last burst is located, take its peak within this span before it
SEARCH_S = 6.0           # search this far before break-off (covers drogue-then-main)
END_MARGIN_S = 0.5       # ...stopping this short of it (skip the turn; exclude the operator's own deploy)
STEP_S = 0.25
ANALYZE_HEIGHT = 480


def _line_columns(path: str, t: float, tmp: str) -> float:
    """Number of columns holding a bright thin vertical line (a suspension line)."""
    subprocess.run(["ffmpeg", "-y", "-ss", f"{max(t, 0):.2f}", "-i", path,
                    "-frames:v", "1", "-vf", f"scale=-2:{ANALYZE_HEIGHT}", tmp],
                   check=False, capture_output=True)
    if not os.path.exists(tmp):
        return 0.0
    g = np.asarray(Image.open(tmp).convert("L"), dtype=np.float32)
    h, w = g.shape
    band = g[int(h * CY0):int(h * CY1), int(w * CX0):int(w * CX1)]
    if not band.size:
        return 0.0
    left = np.roll(band, RIDGE_D, axis=1)
    right = np.roll(band, -RIDGE_D, axis=1)
    ridge = (band - left > RIDGE_T) & (band - right > RIDGE_T)
    ridge[:, :RIDGE_D] = False
    ridge[:, -RIDGE_D:] = False
    return float((ridge.sum(axis=0) > band.shape[0] * COL_FRAC).sum())


# --- drogue throw (stabilization) ---------------------------------------------
# After exit the tandem tumbles; once the drogue is thrown the pair STABILIZES,
# hanging under it on a line — a tall, thin, steady dark column. So the dark
# object's vertical extent jumps to near-full and holds. This is framing-dependent
# (needs the operator filming the drogue column from roughly below), so it fires
# only when that steady tall column is actually present, and returns None otherwise
# rather than guessing.
DVX0, DVX1, DVY0, DVY1 = 0.30, 0.70, 0.08, 0.85
DROGUE_SEARCH_S = 15.0
DROGUE_VEXT_HIGH = 0.85     # "hanging under the drogue" gives a near-full vertical extent
DROGUE_SUSTAIN_S = 1.5      # ...held this long (tumbling is brief and jittery)


def _dark_vext(path: str, t: float, tmp: str) -> float:
    """Vertical extent of the central dark object (0..1 of the crop height)."""
    subprocess.run(["ffmpeg", "-y", "-ss", f"{max(t, 0):.2f}", "-i", path,
                    "-frames:v", "1", "-vf", f"scale=-2:{ANALYZE_HEIGHT}", tmp],
                   check=False, capture_output=True)
    if not os.path.exists(tmp):
        return 0.0
    g = np.asarray(Image.open(tmp).convert("L"), dtype=np.float32)
    h, w = g.shape
    c = g[int(h * DVY0):int(h * DVY1), int(w * DVX0):int(w * DVX1)]
    if c.size == 0:
        return 0.0
    dark = c < (c.mean() * 0.55)
    ys, _ = np.where(dark)
    if ys.size < 15:
        return 0.0
    return float((np.percentile(ys, 95) - np.percentile(ys, 5)) / c.shape[0])


def detect_drogue(path: str, exit_t: float, search_s: float = DROGUE_SEARCH_S, step_s: float = 0.5):
    """Return (drogue_t, confidence) or None.

    drogue_t = onset of the first run where the dark vertical extent stays high
    (pair hanging under the drogue) for DROGUE_SUSTAIN_S, within (exit, exit+search].
    None when no such steady column appears (framing hides it) — the drogue
    boundary is then left for the human.
    """
    lo, hi = exit_t + 1.0, exit_t + search_s
    ts: list[float] = []
    vs: list[float] = []
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        t = lo
        while t <= hi + 1e-6:
            ts.append(t)
            vs.append(_dark_vext(path, t, tmp))
            t += step_s
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if not ts:
        return None
    sm = [(vs[max(0, i - 1)] + vs[i] + vs[min(len(vs) - 1, i + 1)]) / 3.0 for i in range(len(vs))]
    need = max(1, int(round(DROGUE_SUSTAIN_S / step_s)))
    run = 0
    for i in range(len(sm)):
        if sm[i] >= DROGUE_VEXT_HIGH:
            run += 1
            if run >= need:
                start = i - need + 1
                conf = round(min(0.7, 0.4 + (sm[start] - DROGUE_VEXT_HIGH)), 3)
                return ts[start], conf
        else:
            run = 0
    return None


def detect_deploy(path: str, otvorot_t: float, exit_t: float = 0.0,
                  search_s: float = SEARCH_S, step_s: float = STEP_S):
    """Return (onset_t, confidence) or None.

    onset = the peak of the LAST bright-line burst in [otvorot - search_s,
    otvorot - END_MARGIN_S] — the main deploy (the drogue burst is earlier).
    Returns None when no line burst reaches LINE_BURST_MIN.
    """
    lo = max(otvorot_t - search_s, exit_t + 1.0)
    hi = otvorot_t - END_MARGIN_S
    if hi - lo < 1.0:
        return None

    ts: list[float] = []
    lines: list[float] = []
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        t = lo
        while t <= hi + 1e-6:
            ts.append(t)
            lines.append(_line_columns(path, t, tmp))
            t += step_s
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if not ts:
        return None
    # Latest frame that is part of a deploy-strength line burst (drogue bursts are earlier).
    last = None
    for i in range(len(ts)):
        if lines[i] >= LINE_BURST_MIN:
            last = i
    if last is None:
        return None
    # Peak of that last burst (the fullest fan of lines), looking a little before it.
    back = max(1, int(round(BURST_LOOKBACK_S / step_s)))
    lo_i = max(0, last - back)
    k = max(range(lo_i, last + 1), key=lambda i: lines[i])
    conf = round(max(0.4, min(0.85, 0.4 + lines[k] * 0.04)), 3)
    return ts[k], conf

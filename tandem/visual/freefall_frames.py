"""Per-frame analysis of the free fall with MediaPipe: the pair, the passenger's face.

For frames sampled at ``fps`` in [lo, hi]:
- the tandem pair: MediaPipe EfficientDet-Lite2 person boxes; the pair is the largest
  group of overlapping boxes (other jumpers are separate groups, the operator's own arm
  hugs a frame corner and is dropped). Agrees with RT-DETR on the pair's share of the
  frame at r = 0.98 where it finds the pair, at ~0.2 s per frame on CPU; it does lose
  the pair in some footage (dark suits against bright cloud, shot from below) — the GPU
  pipeline (tandem_classifier-GPU) uses RT-DETR boxes instead;
- faces: MediaPipe Face Landmarker on four overlapping, upscaled tiles of the pair box
  (faces in free fall are small; tiling finds them in 26 of 36 close-up frames vs 17 on
  the whole box). Each face: centre, height as a share of the frame height, smile
  (mean of mouthSmileLeft/Right), jawOpen (scream / laugh) and head yaw. The passenger
  is the lowest face — the instructor is behind and above them;
- optionally the backbone embedding of the pair crop (``embed=True``), whose drift tells
  the pair turning in frame.
Model files live in tandem/visual/models/ and are fetched on first use.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
import tempfile
import urllib.request

import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), "models")
URLS = {
    "efficientdet_lite2.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/"
                                 "efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}
FRAME_W = 1280          # analysis width; faces need the resolution
FACE_MIN_PAIR = 0.08    # look for faces only when the pair fills >= 8 % of the frame
_det = _faces = None


def _model(name: str) -> str:
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        os.makedirs(MODELS, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def _load():
    global _det, _faces
    if _det is None:
        from mediapipe.tasks.python import BaseOptions, vision
        _det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=_model("efficientdet_lite2.tflite")),
            score_threshold=0.3, category_allowlist=["person"], max_results=10))
        _faces = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model("face_landmarker.task")),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            num_faces=3, min_face_detection_confidence=0.3))
    return _det, _faces


def pair_box(boxes, W, H):
    """The pair: the largest group of overlapping person boxes, ignoring a small box cut
    by two frame edges (the operator's own arm). -> (box or None, share of frame)."""
    def edge_hand(b):
        touches = (b[0] <= 2) + (b[1] <= 2) + (b[2] >= W - 2) + (b[3] >= H - 2)
        return touches >= 2 and (b[2] - b[0]) * (b[3] - b[1]) < 0.25 * W * H
    groups = []
    for b in (b for b in boxes if not edge_hand(b)):
        hit = [g for g in groups if any(not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]) for o in g)]
        groups = [g for g in groups if g not in hit] + [[b] + [o for g in hit for o in g]]
    if not groups:
        return None, 0.0
    ub = lambda g: [max(0, min(o[0] for o in g)), max(0, min(o[1] for o in g)),
                    min(W, max(o[2] for o in g)), min(H, max(o[3] for o in g))]
    best = max((ub(g) for g in groups), key=lambda u: (u[2] - u[0]) * (u[3] - u[1]))
    return best, (best[2] - best[0]) * (best[3] - best[1]) / (W * H)


def _mp_image(img):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img.convert("RGB"))))


def faces_in(img, box) -> list[dict]:
    """Faces inside the pair box, found on 2x2 overlapping upscaled tiles."""
    _, fl = _load()
    W, H = img.size
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    found: list[dict] = []
    for i in (0, 1):
        for j in (0, 1):
            c = (x0 + i * w / 2 - 0.1 * w * (i > 0), y0 + j * h / 2 - 0.1 * h * (j > 0),
                 x0 + (i + 1) * w / 2 + 0.1 * w * (i < 1), y0 + (j + 1) * h / 2 + 0.1 * h * (j < 1))
            tile = img.crop(tuple(int(v) for v in c))
            s = 640 / max(tile.size)
            tile = tile.resize((max(1, int(tile.size[0] * s)), max(1, int(tile.size[1] * s))))
            r = fl.detect(_mp_image(tile))
            for lm, bs, mat in zip(r.face_landmarks, r.face_blendshapes, r.facial_transformation_matrixes):
                ys = [p.y for p in lm]; xs = [p.x for p in lm]
                cx = c[0] + np.mean(xs) * (c[2] - c[0]); cy = c[1] + np.mean(ys) * (c[3] - c[1])
                fh = (max(ys) - min(ys)) * (c[3] - c[1]) / H
                if any(abs(cx - f["cx"] * W) < 0.08 * w and abs(cy - f["cy"] * H) < 0.08 * h for f in found):
                    continue
                b = {k.category_name: k.score for k in bs}
                R = np.asarray(mat)[:3, :3]
                found.append({"cx": float(cx / W), "cy": float(cy / H), "h": float(fh),
                              "smile": float((b["mouthSmileLeft"] + b["mouthSmileRight"]) / 2),
                              "jaw": float(b["jawOpen"]),
                              "yaw": float(math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))))})
    return found


def analyze(path: str, lo: float, hi: float, fps: float = 2.0, embed: bool = False) -> list[dict]:
    """One dict per sampled frame: t, frac (pair share), box (normalised), faces,
    passenger (the lowest face or None) and, with ``embed``, emb (pair-crop embedding)."""
    from PIL import Image
    det, _ = _load()
    td = tempfile.mkdtemp()
    out: list[dict] = []
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{lo:.2f}", "-to", f"{hi:.2f}", "-i", path,
                        "-vf", f"fps={fps},scale={FRAME_W}:-2", os.path.join(td, "f_%04d.jpg")], check=False)
        frames = sorted(glob.glob(os.path.join(td, "f_*.jpg")))
        crops = []
        for k, fp in enumerate(frames):
            img = Image.open(fp).convert("RGB"); W, H = img.size
            r = det.detect(_mp_image(img))
            boxes = [[d.bounding_box.origin_x, d.bounding_box.origin_y,
                      d.bounding_box.origin_x + d.bounding_box.width,
                      d.bounding_box.origin_y + d.bounding_box.height] for d in r.detections]
            box, frac = pair_box(boxes, W, H)
            faces = faces_in(img, box) if box and frac >= FACE_MIN_PAIR else []
            rec = {"t": round(lo + k / fps, 3), "frac": round(frac, 4),
                   "box": [round(box[0] / W, 4), round(box[1] / H, 4), round(box[2] / W, 4), round(box[3] / H, 4)] if box else None,
                   "faces": faces, "passenger": max(faces, key=lambda f: f["cy"]) if faces else None}
            if embed and box:
                cp = fp.replace(".jpg", "_c.jpg"); img.crop(tuple(int(v) for v in box)).resize((224, 224)).save(cp)
                crops.append((len(out), cp))
            out.append(rec)
        if embed and crops:
            from tandem.visual import probe
            _, model = probe._load()
            emb = probe._embed([c for _, c in crops], model)
            for (i, _), e in zip(crops, emb):
                out[i]["emb"] = e
    finally:
        for fp in glob.glob(os.path.join(td, "*")):
            os.remove(fp)
        os.rmdir(td)
    return out

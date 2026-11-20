"""Synthetic test data: drawn 'people' in flat colours + a colour-based fake SAM 3.

Used only by the offline tests; nothing here touches real images or the network.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

COLORS = {  # part -> RGB used to draw it
    "skin": (224, 172, 140), "top": (200, 30, 30), "bottom": (30, 60, 200),
    "dress": (120, 40, 160), "hat": (250, 200, 0), "shoe": (20, 140, 70), "apron": (250, 120, 180),
}


def person_image(seed: int, w: int = 600, h: int = 900, outfit: str = "two_piece",
                 second_person: bool = False, tiny: bool = False) -> Image.Image:
    rng = np.random.default_rng(seed)
    bg = rng.integers(90, 140, size=(h, w, 3)).astype(np.uint8)          # textured, not white
    bg[..., 2] = np.clip(bg[..., 2] + 30, 0, 255)                       # bluish grey: no part colour nearby
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    for _ in range(4):                                                    # grey clutter so photos differ
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        v = int(rng.integers(20, 200))
        d.rectangle([x0, y0, x0 + int(rng.integers(40, 300)), y0 + int(rng.integers(40, 300))], fill=(v, v, v))
    s = (0.25 if tiny else 1.0) * float(rng.uniform(0.85, 1.1))
    sx = float(rng.uniform(0.7, 1.3))                                      # body width varies
    cx = w // 2 + int(rng.integers(-40, 40))
    top = int(h * 0.08)

    def P(x, y):
        return cx + x * s * sx, top + y * s

    def rect(x0, y0, x1, y1, col):
        d.rectangle([*P(x0, y0), *P(x1, y1)], fill=COLORS[col])

    d.ellipse([*P(-45, 60), *P(45, 160)], fill=COLORS["skin"])           # head
    rect(-80, 0, 80, 70, "hat")                                           # hat on head
    if outfit == "two_piece":
        rect(-110, 160, 110, 420, "top")
        rect(-120, 420, 120, 700, "bottom")
        rect(-60, 380, 60, 520, "apron")
    else:
        rect(-110, 160, 110, 700, "dress")
    rect(-90, 700, -20, 740, "shoe")
    rect(20, 700, 90, 740, "shoe")
    rect(-140, 170, -110, 400, "skin")                                    # arms
    rect(110, 170, 140, 400, "skin")
    # fabric texture: small noise inside every drawn part (colour detection still works)
    a = np.asarray(im).astype(np.int16)
    a += rng.integers(-12, 13, size=a.shape, dtype=np.int16)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    if second_person:
        d.rectangle([5, top + 150, 105, top + 700], fill=COLORS["top"])      # clear of the main person
    return im


def jpeg(im: Image.Image, q: int = 92) -> bytes:
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return b.getvalue()


def _near(a: np.ndarray, col, tol: int = 40) -> np.ndarray:
    return (np.abs(a.astype(int) - np.array(col)).sum(axis=2) < tol)


class ColorPredictor:
    """Pretends to be Sam3Prompter: finds parts by their drawing colour."""

    PROMPT_PART = {"jacket": "top", "blouse": "top", "shirt": "top", "skirt": "bottom", "trousers": "bottom",
                   "dress": "dress", "robe": "dress", "hat": "hat", "headdress": "hat", "shoe": "shoe",
                   "boot": "shoe", "apron": "apron", "belt": "apron"}

    def __init__(self, dress_also_on_two_piece: bool = True):
        self.dress_also = dress_also_on_two_piece
        self.calls = 0

    def __call__(self, image, prompts):
        self.calls += 1
        a = np.asarray(image.convert("RGB"))
        parts = {k: _near(a, v) for k, v in COLORS.items()}
        out = {}
        for p in prompts:
            if p == "person":
                body = np.zeros(a.shape[:2], bool)
                for k in ("skin", "top", "bottom", "dress", "shoe", "apron"):
                    body |= parts[k]
                # split into connected blobs by columns: main body vs. far-left second person
                out[p] = _blobs(body, 0.9)
            elif p in ("clothing", "garment"):
                out[p] = []
            elif p in self.PROMPT_PART:
                part = self.PROMPT_PART[p]
                m = parts[part]
                if part == "dress" and self.dress_also and parts["top"].any():
                    m = parts["top"] | parts["bottom"]            # SAM-style confusion on two-piece outfits
                out[p] = [(m, 0.8)] if m.sum() > 50 else []
            else:
                out[p] = []
        return out


def _blobs(mask: np.ndarray, score: float):
    cols = mask.any(axis=0)
    blobs, start = [], None
    for x, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = x
        elif not v and start is not None:
            m = np.zeros_like(mask)
            m[:, start:x] = mask[:, start:x]
            if m.sum() > 200:
                blobs.append(m)
            start = None
    blobs.sort(key=lambda m: -m.sum())
    return [(m, score - 0.05 * i) for i, m in enumerate(blobs)]

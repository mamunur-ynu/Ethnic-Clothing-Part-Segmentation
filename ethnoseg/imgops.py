"""Small image helpers shared by screen / compose / qc (numpy + Pillow + OpenCV)."""
from __future__ import annotations

import numpy as np
from PIL import Image

WHITE = (255, 255, 255)


# --------------------------------------------------------------------------- hashing
def dhash(img: Image.Image, size: int = 8) -> str:
    """Difference hash with size*size bits as hex (robust to resizing / recompression).

    size=8 -> 64 bits (16 hex chars), size=16 -> 256 bits (64 hex chars).
    """
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def phash(img: Image.Image) -> str:
    """64-bit DCT perceptual hash (16 hex chars). Stable under rescaling, JPEG and small crops."""
    import cv2

    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(g)[:8, :8].flatten()
    bits = low[1:] > np.median(low[1:])                     # skip the DC term
    bits = np.concatenate([[False], bits])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def subject_hash(img: np.ndarray, subject: np.ndarray) -> str:
    """pHash of the subject's bounding-box crop (ignores the white margin around it)."""
    b = bbox(subject)
    crop = img if b is None else img[b[1]:b[3], b[0]:b[2]]
    return phash(Image.fromarray(crop))


def subject_thumb(img: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """48x64 RGB thumbnail of the subject crop (float32 0..1), for pixel-level duplicate checks."""
    b = bbox(subject)
    crop = img if b is None else img[b[1]:b[3], b[0]:b[2]]
    return np.asarray(Image.fromarray(crop).resize((48, 64), Image.Resampling.BOX), np.float32) / 255


def thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _words(hashes: list[str]) -> np.ndarray:
    """hex hashes (same length, multiple of 16 chars) -> (n, k) uint64."""
    k = len(hashes[0]) // 16
    return np.array([[int(h[16 * w:16 * w + 16], 16) for w in range(k)] for h in hashes], dtype=np.uint64)


def hamming_pairs(hashes: list[str], max_bits: int, block: int = 256):
    """Yield (i, j, dist) with i < j and Hamming distance <= max_bits, in row blocks."""
    if not hashes:
        return
    v = _words(hashes)
    n, k = v.shape
    for s in range(0, n, block):
        x = v[s:s + block, None, :] ^ v[None, :, :]                        # (b, n, k)
        d = _POPCOUNT[x.view(np.uint8).reshape(x.shape[0], n, k * 8)].sum(axis=2)
        ii, jj = np.nonzero(d <= max_bits)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i + s < j:
                yield i + s, j, int(d[i, j])


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def greedy_duplicates(hashes: list[str], priority: list[int], max_bits: int, confirm=None) -> dict[int, int]:
    """Non-chaining de-duplication.

    Images are visited in `priority` order (best first). An image is a duplicate only if
    it is within max_bits of an image that was already KEPT (and confirm(i, j) agrees,
    when given); otherwise it is kept. Returns {duplicate_index: kept_index}.
    A~B and B~C never removes C because of A.
    """
    nbrs: dict[int, list[tuple[int, int]]] = {}
    for i, j, dist in hamming_pairs(hashes, max_bits):
        if confirm is not None and not confirm(i, j):
            continue
        nbrs.setdefault(i, []).append((dist, j))
        nbrs.setdefault(j, []).append((dist, i))
    kept: set[int] = set()
    dup: dict[int, int] = {}
    for i in priority:
        close = sorted(x for x in nbrs.get(i, []) if x[1] in kept)
        if close:
            dup[i] = close[0][1]
        else:
            kept.add(i)
    return dup


# --------------------------------------------------------------------------- quality
def blur_score(img: Image.Image, side: int = 512) -> float:
    """Variance of the Laplacian on a grayscale copy scaled to `side` px (higher = sharper)."""
    import cv2

    g = img.convert("L")
    s = side / max(g.size)
    if s < 1:
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.Resampling.LANCZOS)
    return float(cv2.Laplacian(np.asarray(g), cv2.CV_64F).var())


# --------------------------------------------------------------------------- geometry
def work_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """Size SAM 3 works at. compose must use the very same formula."""
    s = min(1.0, max_side / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) with x1/y1 exclusive, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_on_white(img: np.ndarray, label: np.ndarray, subject: np.ndarray, size: int, fill: float):
    """Crop to the subject, scale its longer side to fill*size, centre on a white square.

    img: HxWx3 uint8, label: HxW uint8 class map, subject: HxW bool.
    Returns (image, label, subject) at size x size. Background pixels are exactly 255.
    """
    x0, y0, x1, y1 = bbox(subject)
    crop_img, crop_lab, crop_sub = img[y0:y1, x0:x1], label[y0:y1, x0:x1], subject[y0:y1, x0:x1]
    h, w = crop_sub.shape
    s = fill * size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = Image.fromarray(crop_img).resize((nw, nh), Image.Resampling.LANCZOS)
    lab = Image.fromarray(crop_lab).resize((nw, nh), Image.Resampling.NEAREST)
    sub = Image.fromarray(crop_sub.astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out_img = np.full((size, size, 3), 255, np.uint8)
    out_lab = np.zeros((size, size), np.uint8)
    out_sub = np.zeros((size, size), bool)
    out_img[oy:oy + nh, ox:ox + nw] = np.asarray(im)
    out_lab[oy:oy + nh, ox:ox + nw] = np.asarray(lab)
    out_sub[oy:oy + nh, ox:ox + nw] = np.asarray(sub) > 127
    out_img[~out_sub] = 255                    # pure white outside the subject
    out_lab[~out_sub] = 0                      # labels only on the subject
    return out_img, out_lab, out_sub


def placement(subject: np.ndarray) -> tuple[float, float, float]:
    """(dx, dy, fill): centre offsets as fractions of the side, bbox longer side / image side."""
    h, w = subject.shape
    b = bbox(subject)
    if b is None:
        return 1.0, 1.0, 0.0
    x0, y0, x1, y1 = b
    dx = abs((x0 + x1) / 2 - w / 2) / w
    dy = abs((y0 + y1) / 2 - h / 2) / h
    fill = max(x1 - x0, y1 - y0) / max(w, h)
    return dx, dy, fill


# --------------------------------------------------------------------------- visualisation
PALETTE = np.array([
    [255, 255, 255],   # 0 background
    [230, 57, 70],     # 1 top        red
    [29, 120, 230],    # 2 bottom     blue
    [142, 68, 173],    # 3 dress      purple
    [255, 183, 3],     # 4 headwear   amber
    [33, 158, 107],    # 5 footwear   green
    [255, 112, 166],   # 6 waistwear  pink
] + [[90, 90, 90]] * 249, dtype=np.uint8)


def colorize(label: np.ndarray) -> np.ndarray:
    return PALETTE[label]


def overlay(img: np.ndarray, label: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    col = colorize(label).astype(np.float32)
    out = img.astype(np.float32)
    m = label > 0
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    return out.astype(np.uint8)

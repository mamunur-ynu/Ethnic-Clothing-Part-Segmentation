"""Turn SAM 3 instance masks into one subject mask and one part-label map.

Pure numpy; the SAM 3 calls live in sam_backend.py. Input is a dict
    prompt -> list of (mask: HxW bool, score: float)
Output label map (uint8, HxW):
    0         outside the subject
    1..N      clothing part class ids (config 'classes')
    255       subject pixel that is not clothing (skin, hair, hands, props on the body)
compose.py later turns 255 into 0, so the final masks only contain 0..N.

Rules (documented in the dataset README):
  1. Subject = the highest-scoring 'person' instance. Any other person at least
     `second_subject_ratio` of its size (and not the same person) -> multiple_subjects.
     No person -> largest 'clothing'/'garment' instance (garment-only photos).
  2. A part instance is used if score >= part_threshold and at least
     `part_min_inside_subject` of it overlaps the subject's own mask, slightly dilated
     (`part_min_inside_loose` for headwear/footwear, which often stick out of the person
     mask). Parts of a neighbouring person or mannequin therefore never attach.
     Accepted parts are added to the subject (e.g. tall headdresses the person mask misses).
  3. Classes are painted in `paint_order`; later classes win where masks overlap,
     so specific small parts (headwear, waistwear, footwear) are drawn over large ones.
  4. Two-piece rule: if top AND bottom each cover >= two_piece_min_share of the subject,
     'dress' pixels not covered by them are reassigned to top or bottom (whichever
     centroid is vertically closer); if both are absent or small, 'dress' is kept.
  5. Only pixels inside the final subject are labelled.
"""
from __future__ import annotations

import numpy as np

Instance = tuple[np.ndarray, float]
RULES_VERSION = 2          # bump when the rules change; older SAM 3 results are then redone
LOOSE_CLASSES = ("headwear", "footwear")


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r pixels (numpy only, via a 2-D running sum)."""
    if r <= 0:
        return mask.copy()
    h, w = mask.shape
    c = np.pad(mask.astype(np.int32), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    s = c[2 * r + 1:, 2 * r + 1:] - c[:h, 2 * r + 1:] - c[2 * r + 1:, :w] + c[:h, :w]
    return s > 0


def _area(m: np.ndarray) -> int:
    return int(m.sum())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _box_mask(mask: np.ndarray, grow: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if len(xs) == 0:
        return out
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    gx, gy = int((x1 - x0) * grow), int((y1 - y0) * grow)
    out[max(0, y0 - gy):min(h, y1 + gy), max(0, x0 - gx):min(w, x1 + gx)] = True
    return out


def pick_subject(persons: list[Instance], garments: list[Instance], cfg: dict):
    """Returns (subject_mask or None, info dict)."""
    persons = sorted([p for p in persons if p[1] >= cfg["person_threshold"]], key=lambda p: -p[1])
    info = {"n_persons": 0, "subject_kind": None, "subject_score": 0.0}
    if persons:
        main, score = persons[0]
        distinct = [main]
        for m, s in persons[1:]:
            inside = np.logical_and(m, main).sum() / max(1, _area(m))
            if _iou(m, main) < 0.5 and inside < 0.8 and all(_iou(m, d) < 0.5 for d in distinct):
                distinct.append(m)
        info.update(n_persons=len(distinct), subject_kind="person", subject_score=round(score, 4))
        big_others = [d for d in distinct[1:] if _area(d) >= cfg["second_subject_ratio"] * _area(main)]
        if big_others:
            info["status"] = "multiple_subjects"
            return None, info
        return main.copy(), info
    garments = [g for g in garments if g[1] >= cfg["part_threshold"]]
    if garments:
        m, s = max(garments, key=lambda g: _area(g[0]))
        info.update(subject_kind="garment", subject_score=round(s, 4))
        return m.copy(), info
    info["status"] = "no_subject"
    return None, info


def build_label_map(subject: np.ndarray, part_hits: dict[str, list[Instance]],
                    classes: list[dict], paint_order: list[str], cfg: dict):
    """part_hits: class name -> instances (already merged over that class's prompts)."""
    ids = {c["name"]: c["id"] for c in classes}
    region = _box_mask(subject, 0.10)
    ys, xs = np.nonzero(subject)
    r = max(2, int(0.03 * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))) if len(ys) else 2
    anchor = dilate(subject, r)
    accepted: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for name in paint_order:
        layer = np.zeros_like(subject, dtype=bool)
        need = cfg.get("part_min_inside_loose", 0.05) if name in LOOSE_CLASSES else cfg["part_min_inside_subject"]
        for m, s in part_hits.get(name, []):
            if s < cfg["part_threshold"] or _area(m) == 0:
                continue
            if np.logical_and(m, anchor).sum() / _area(m) < need:
                continue
            layer |= m
            scores[name] = max(scores.get(name, 0.0), float(s))
        if layer.any():
            accepted[name] = layer

    full_subject = subject.copy()
    for layer in accepted.values():
        full_subject |= layer & region

    label = np.zeros(subject.shape, np.uint8)
    label[full_subject] = 255
    for name in paint_order:
        if name in accepted:
            label[accepted[name] & full_subject] = ids[name]

    # two-piece rule
    subj_px = max(1, int(full_subject.sum()))
    t, b, d = ids.get("top"), ids.get("bottom"), ids.get("dress")
    two_piece = False
    if t is not None and b is not None and d is not None:
        top_m, bot_m = label == t, label == b
        if top_m.sum() >= cfg["two_piece_min_share"] * subj_px and bot_m.sum() >= cfg["two_piece_min_share"] * subj_px:
            two_piece = True
            dress_m = label == d
            if dress_m.any():
                yt, yb = np.nonzero(top_m)[0].mean(), np.nonzero(bot_m)[0].mean()
                rows = np.nonzero(dress_m)
                to_top = np.abs(rows[0] - yt) <= np.abs(rows[0] - yb)
                label[rows[0][to_top], rows[1][to_top]] = t
                label[rows[0][~to_top], rows[1][~to_top]] = b
        elif (label == d).sum() < cfg["leftover_dress_min_share"] * subj_px:
            label[label == d] = 255

    share = {name: round(float((label == ids[name]).sum()) / subj_px, 4) for name in ids if name != "background"}
    summary = {
        "subject_px": subj_px,
        "clothing_share": round(float(((label > 0) & (label < 255)).sum()) / subj_px, 4),
        "part_share": share,
        "part_score": {k: round(v, 4) for k, v in scores.items()},
        "two_piece": two_piece,
        "rules": RULES_VERSION,
    }
    return label, summary


def process_file(predict, src_path, out_dir, image_id: str, classes: list[dict],
                 paint_order: list[str], cfg: dict) -> dict:
    """Resize one raw image to the working size, segment it, write <ID>.png + <ID>.json.

    This is exactly what the Modal GPU job runs for every image.
    """
    import json
    from pathlib import Path

    from PIL import Image

    from .imgops import work_size

    out_dir = Path(out_dir)
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            size = work_size(im.width, im.height, cfg["work_max_side"])
            if size != im.size:
                im = im.resize(size, Image.Resampling.LANCZOS)
        label, summary = segment_with(predict, im, classes, paint_order, cfg)
        summary.update(image_id=image_id, work_size=list(size))
        if label is not None:
            Image.fromarray(label).save(out_dir / f"{image_id}.png")
    except Exception as e:                       # keep going; the failure is reported per image
        summary = {"image_id": image_id, "status": "error", "error": repr(e)[:300]}
    (out_dir / f"{image_id}.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def segment_with(predict, image, classes: list[dict], paint_order: list[str], cfg: dict):
    """Full rule set given predict(image, prompts) -> {prompt: [(mask, score), ...]}.

    Returns (label_map or None, summary dict with 'status').
    """
    prompts = [cfg["person_prompt"], *cfg["garment_fallback_prompts"]]
    part_prompts = {c["name"]: list(c.get("prompts", [])) for c in classes if c["id"] != 0}
    for ps in part_prompts.values():
        prompts.extend(ps)
    hits = predict(image, prompts)
    garments = [inst for p in cfg["garment_fallback_prompts"] for inst in hits.get(p, [])]
    subject, info = pick_subject(hits.get(cfg["person_prompt"], []), garments, cfg)
    if subject is None:
        info["rules"] = RULES_VERSION
        return None, info
    per_class = {name: [inst for p in ps for inst in hits.get(p, [])] for name, ps in part_prompts.items()}
    label, summary = build_label_map(subject, per_class, classes, paint_order, cfg)
    summary.update(info)
    summary["status"] = "ok"
    return label, summary

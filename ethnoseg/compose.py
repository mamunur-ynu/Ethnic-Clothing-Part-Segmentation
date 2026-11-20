"""Stage 4: build the final 512x512 white-background image and mask from SAM 3 output.

For every 'segmented' image (or 'screened' with a seg/<ID>.json present):
    work/img/<ID>.png      RGB, pure white (255,255,255) outside the subject
    work/mask/<ID>.png     uint8 class ids 0..6 (same size, same name)
    work/subject/<ID>.png  subject mask (used by QC to prove the background is white)
    work/preview/<ID>.jpg  side-by-side preview for the reviewer

Automatic rejections: error, no_subject, multiple_subjects, subject_too_small, no_parts,
near_duplicate_final (two cut-outs that look the same, even if the raw photos differed).
Warnings for the reviewer (flags): touches_top, touches_bottom, touches_side, low_clothing.
PNG is used for images because JPEG compression would put non-white pixels around the subject.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import imgops
from .manifest import Manifest
from .settings import Settings


def drop_specks(lab: np.ndarray, min_share: float) -> np.ndarray:
    """Remove mask fragments smaller than min_share of the largest one.

    SAM 3 masks sometimes contain small stray blobs far from the person; left in, they
    would stretch the subject's bounding box and break centring and scaling.
    """
    import cv2

    n, comp, stats, _ = cv2.connectedComponentsWithStats((lab > 0).astype(np.uint8), connectivity=8)
    if n <= 2:
        return lab
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= min_share * areas.max()
    out = lab.copy()
    out[~keep[comp]] = 0
    return out


def garment_type(label: np.ndarray, classes: list[dict], min_share: float = 0.01) -> str:
    total = max(1, int((label > 0).sum()))
    present = [c["name"] for c in classes if c["id"] and (label == c["id"]).sum() / total >= min_share]
    return "+".join(present) if present else "none"


def compose_one(S: Settings, row) -> dict:
    cfg, seg_cfg = S.section("compose"), S.section("segment")
    summary = json.loads((S.seg_dir / f"{row['image_id']}.json").read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        return {"reject": summary.get("status", "error"), "seg_json": json.dumps(summary)}
    lab = np.asarray(Image.open(S.seg_dir / f"{row['image_id']}.png"))
    with Image.open(S.raw_dir / row["file"]) as im:
        im = im.convert("RGB")
        size = imgops.work_size(im.width, im.height, seg_cfg["work_max_side"])
        if size != im.size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        img = np.asarray(im)
    if lab.shape != img.shape[:2]:
        return {"reject": "size_mismatch", "seg_json": json.dumps(summary)}

    lab = drop_specks(lab, cfg.get("min_fragment_share", 0.02))
    subject = lab > 0
    x0, y0, x1, y1 = imgops.bbox(subject)
    flags = []
    if max(x1 - x0, y1 - y0) < cfg["min_subject_px"]:
        return {"reject": "subject_too_small", "seg_json": json.dumps(summary)}
    h, w = subject.shape
    edge = 2
    if y0 < edge:
        flags.append("touches_top")
    if y1 > h - edge:
        flags.append("touches_bottom")
    if x0 < edge or x1 > w - edge:
        flags.append("touches_side")
    clothing = ((lab > 0) & (lab < 255)).sum() / max(1, subject.sum())
    if clothing < cfg["min_clothing_share"]:
        return {"reject": "no_parts", "seg_json": json.dumps(summary)}
    if clothing < 0.35:
        flags.append("low_clothing")

    final_lab = np.where(lab == 255, 0, lab).astype(np.uint8)
    out_img, out_lab, out_sub = imgops.fit_on_white(img, final_lab, subject, cfg["size"], cfg["fill"])
    dx, dy, fill = imgops.placement(out_sub)

    wd = S.work_dir
    for sub in ("img", "mask", "subject", "preview"):
        (wd / sub).mkdir(parents=True, exist_ok=True)
    iid = row["image_id"]
    Image.fromarray(out_img).save(wd / "img" / f"{iid}.png", optimize=True)
    Image.fromarray(out_lab).save(wd / "mask" / f"{iid}.png", optimize=True)
    Image.fromarray(out_sub.astype(np.uint8) * 255).save(wd / "subject" / f"{iid}.png", optimize=True)
    prev = np.concatenate([out_img, imgops.overlay(out_img, out_lab)], axis=1)
    Image.fromarray(prev).save(wd / "preview" / f"{iid}.jpg", quality=88)
    return {
        "seg_json": json.dumps(summary),
        "center_dx": round(dx, 4), "center_dy": round(dy, 4), "fill": round(fill, 4),
        "flags": ",".join(flags), "garment_type": garment_type(out_lab, S.classes),
        "final_dhash": imgops.subject_hash(out_img, out_sub),
    }


def run(S: Settings, redo: bool = False, progress=print) -> dict:
    stats: dict[str, int] = {}
    with Manifest(S.db_path) as db:
        statuses = ("screened", "segmented", "composed") if redo else ("screened", "segmented")
        rows = [r for r in db.rows(f"status IN ({','.join('?' * len(statuses))})", statuses)
                if (S.seg_dir / f"{r['image_id']}.json").exists()]
        with db.tx():
            for i, r in enumerate(rows, 1):
                res = compose_one(S, r)
                reason = res.pop("reject", None)
                db.update(r["image_id"], **res)
                if reason:
                    db.reject(r["image_id"], reason)
                else:
                    db.update(r["image_id"], status="composed")
                stats[reason or "composed"] = stats.get(reason or "composed", 0) + 1
                if i % 200 == 0:
                    progress(f"  composed {i}/{len(rows)}")

        # cut-outs that look alike (same model and costume, photo cropped differently)
        cand = db.rows("status IN ('composed','kept') AND final_dhash IS NOT NULL")
        order = sorted(range(len(cand)), key=lambda i: (cand[i]["status"] != "kept",
                                                         -(cand[i]["width"] * cand[i]["height"]),
                                                         cand[i]["image_id"]))
        thumbs: dict[int, np.ndarray] = {}

        def thumb(i):
            if i not in thumbs:
                iid = cand[i]["image_id"]
                im = np.asarray(Image.open(S.work_dir / "img" / f"{iid}.png").convert("RGB"))
                sub = np.asarray(Image.open(S.work_dir / "subject" / f"{iid}.png")) > 127
                thumbs[i] = imgops.subject_thumb(im, sub)
            return thumbs[i]

        cc = S.section("compose")
        dups = imgops.greedy_duplicates(
            [r["final_dhash"] for r in cand], order, cc["final_near_dup_hamming"],
            confirm=lambda i, j: imgops.thumb_diff(thumb(i), thumb(j)) <= cc["final_pixel_diff"])
        with db.tx():
            for i, k in dups.items():
                a, b = cand[i], cand[k]
                if a["grp"] != b["grp"]:
                    db.reject(a["image_id"], "conflicting_labels")
                    db.reject(b["image_id"], "conflicting_labels")
                    stats["conflicting_labels"] = stats.get("conflicting_labels", 0) + 2
                else:
                    db.reject(a["image_id"], f"near_duplicate_final:{b['image_id']}")
                    stats["near_duplicate_final"] = stats.get("near_duplicate_final", 0) + 1
            db.log("compose", str(stats))
    progress(f"compose: {stats}")
    return stats

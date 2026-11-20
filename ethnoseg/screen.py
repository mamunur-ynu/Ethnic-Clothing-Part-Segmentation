"""Stage 2: per-image checks, then near-duplicate removal across the whole collection.

Per image (status 'candidate'):
    corrupt          cannot be decoded
    too_small        shorter side < screen.min_side_px
    bad_aspect       longer/shorter side > screen.max_aspect (banners, strip collages)
    blurry           variance of Laplacian < screen.blur_min
Then over every image still in the pipeline:
    near_duplicate_of:<id>   64-bit pHash distance <= screen.near_dup_hamming to an image that is
                             kept (no chaining); the one kept is the one furthest along, then the largest
    conflicting_labels       the same picture saved under two different groups (both removed)
"""
from __future__ import annotations

from PIL import Image

from . import imgops
from .manifest import Manifest
from .settings import Settings

ACTIVE = ("screened", "segmented", "composed", "kept")
STAGE_RANK = {"screened": 0, "segmented": 1, "composed": 2, "kept": 3}


def check_image(path, cfg) -> dict:
    try:
        with Image.open(path) as im:
            im.load()
            im = im.convert("RGB")
    except Exception:
        return {"reason": "corrupt"}
    w, h = im.size
    out = {"width": w, "height": h, "dhash": imgops.phash(im), "blur": imgops.blur_score(im)}
    if min(w, h) < cfg["min_side_px"]:
        out["reason"] = "too_small"
    elif max(w, h) / min(w, h) > cfg["max_aspect"]:
        out["reason"] = "bad_aspect"
    elif out["blur"] < cfg["blur_min"]:
        out["reason"] = "blurry"
    return out


def run(S: Settings, progress=print) -> dict:
    cfg = S.section("screen")
    stats: dict[str, int] = {}
    with Manifest(S.db_path) as db:
        todo = db.rows("status='candidate'")
        with db.tx():
            for i, r in enumerate(todo, 1):
                res = check_image(S.raw_dir / r["file"], cfg)
                reason = res.pop("reason", None)
                db.update(r["image_id"], **res)
                if reason:
                    db.reject(r["image_id"], reason)
                else:
                    db.update(r["image_id"], status="screened")
                stats[reason or "passed"] = stats.get(reason or "passed", 0) + 1
                if i % 250 == 0:
                    progress(f"  screened {i}/{len(todo)}")

        rows = db.rows(f"status IN ({','.join('?' * len(ACTIVE))}) AND dhash IS NOT NULL", ACTIVE)
        order = sorted(range(len(rows)), key=lambda i: (-STAGE_RANK[rows[i]["status"]],
                                                         -(rows[i]["width"] * rows[i]["height"]),
                                                         rows[i]["image_id"]))
        dups = imgops.greedy_duplicates([r["dhash"] for r in rows], order, cfg["near_dup_hamming"])
        with db.tx():
            for i, k in dups.items():
                a, b = rows[i], rows[k]
                if a["grp"] != b["grp"]:          # same picture saved under two groups: label is doubtful
                    db.reject(a["image_id"], "conflicting_labels")
                    db.reject(b["image_id"], "conflicting_labels")
                    stats["conflicting_labels"] = stats.get("conflicting_labels", 0) + 2
                else:
                    db.reject(a["image_id"], f"near_duplicate_of:{b['image_id']}")
                    stats["near_duplicate"] = stats.get("near_duplicate", 0) + 1
            db.log("screen", str(stats))
    progress(f"screen: {stats}")
    return stats

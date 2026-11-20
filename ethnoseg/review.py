"""Stage 5: human review of every composed image (python -m ethnoseg review --reviewer NAME).

Opens an OpenCV window showing the white-background image, the coloured mask on top
of it, the class legend and the source evidence. Keys:

    K or Enter   keep
    R            reject, then press the reason number (1-7)
    B            back to the previous image (its decision can be changed)
    S            skip for now
    Q or Esc     quit (decisions are saved immediately; run again to continue)

Only 'kept' images go into the final dataset. This is the manual check of the
automatic SAM 3 masks that the brief requires, so it must be done by a person.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import imgops
from .manifest import Manifest
from .settings import Settings

REASONS = {
    "1": "wrong_parts",          # top/bottom/dress/headwear mislabelled or a big part missing
    "2": "bad_cutout",           # background left in, or body/garment cut off by the mask
    "3": "multiple_subjects",
    "4": "incomplete_outfit",    # cropped body, heavy occlusion
    "5": "not_real_photo",       # AI-generated, drawing, mannequin collage, poster text
    "6": "label_doubt",          # evidence does not really support the group label
    "7": "other",
}
FONT_CANDIDATES = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                   "/System/Library/Fonts/PingFang.ttc"]


def _font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render(S: Settings, row, index: int, total: int, pending_reject: bool = False) -> np.ndarray:
    """Build the review canvas (RGB uint8) for one image."""
    img = np.asarray(Image.open(S.work_dir / "img" / f"{row['image_id']}.png").convert("RGB"))
    lab = np.asarray(Image.open(S.work_dir / "mask" / f"{row['image_id']}.png"))
    left = np.concatenate([img, imgops.overlay(img, lab)], axis=1)
    panel = Image.new("RGB", (380, left.shape[0]), (245, 245, 245))
    d = ImageDraw.Draw(panel)
    f, fs = _font(18), _font(15)
    g = S.group(row["grp"])
    y = 12
    d.text((12, y), f"{index + 1}/{total}   {row['image_id']}", fill=(0, 0, 0), font=f); y += 30
    d.text((12, y), f"Group: {g['name']} {g['zh']}", fill=(0, 0, 0), font=f); y += 28
    d.text((12, y), f"Source check: {row['verify_status']}", fill=(60, 60, 60), font=fs); y += 22
    ev = (row["evidence"] or "")[:90]
    for k in range(0, len(ev), 22):
        d.text((12, y), ev[k:k + 22], fill=(60, 60, 60), font=fs); y += 20
    y += 8
    if row["flags"]:
        d.text((12, y), f"Warnings: {row['flags']}", fill=(200, 60, 0), font=fs); y += 26
    if row["review"]:
        d.text((12, y), f"Current decision: {row['review']} {row['review_reason'] or ''}",
               fill=(0, 90, 160), font=fs); y += 22
        d.text((12, y), f"  by {row['reviewer'] or '?'}", fill=(0, 90, 160), font=fs); y += 26
    y += 6
    for c in S.part_classes:
        d.rectangle((12, y + 3, 28, y + 17), fill=tuple(int(v) for v in imgops.PALETTE[c["id"]]))
        d.text((36, y), f"{c['id']} {c['name']}", fill=(0, 0, 0), font=fs); y += 22
    y += 10
    if pending_reject:
        d.text((12, y), "Reject reason:", fill=(200, 0, 0), font=f); y += 26
        for k, v in REASONS.items():
            d.text((12, y), f"{k}  {v}", fill=(200, 0, 0), font=fs); y += 20
    else:
        for line in ("K/Enter keep    R reject", "B back    S skip    Q quit"):
            d.text((12, y), line, fill=(0, 0, 0), font=fs); y += 22
    return np.concatenate([left, np.asarray(panel)], axis=1)


def run(S: Settings, reviewer: str, include_done: bool = False, group: str | None = None,
        cv2=None, progress=print, min_view: float = 0.0, clock=time.monotonic) -> dict:
    """Show each image and save the reviewer's keep/reject decision.

    min_view > 0: a decision key only counts if the image has been on screen that long and the
    key is a fresh press (not auto-repeat from a held-down key), so every image is really looked at.
    Decisions are saved immediately; running again continues with the images not yet reviewed."""
    if cv2 is None:
        import cv2  # noqa: F811  (injectable for tests)
    where = "status IN ('composed','kept')" if include_done else "status='composed' AND review IS NULL"
    params: tuple = ()
    if group:
        where += " AND grp=?"
        params = (group.upper(),)
    win = "Review - ethnic clothing masks"
    tally = {"keep": 0, "reject": 0}
    with Manifest(S.db_path) as db:
        rows = db.rows(where, params)
        if not rows:
            progress("Nothing to review.")
            return tally
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        i, pending = 0, False
        shown_i, shown_at, last_key_at = -1, 0.0, -1e9
        decision_keys = set("kr") | set(REASONS)
        while 0 <= i < len(rows):
            row = db.get(rows[i]["image_id"])
            canvas = render(S, row, i, len(rows), pending)
            cv2.imshow(win, canvas[:, :, ::-1])
            if i != shown_i:
                shown_i, shown_at = i, clock()
            key = cv2.waitKey(0) & 0xFF
            ch = chr(key).lower() if key < 128 else ""
            t, prev_key_at, last_key_at = clock(), last_key_at, clock()
            if min_view > 0 and (ch in decision_keys or key in (13, 10)):
                if t - shown_at < min_view or t - prev_key_at < 0.15:   # too soon, or a held key repeating
                    continue
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if pending:
                if ch in REASONS:
                    with db.tx():
                        db.update(row["image_id"], review="reject", review_reason=REASONS[ch],
                                  reviewer=reviewer, reviewed_at=now, status="rejected",
                                  reason=f"review:{REASONS[ch]}")
                    tally["reject"] += 1
                    i += 1
                pending = False                      # any other key cancels
                continue
            if ch in ("q",) or key == 27:
                break
            if ch == "k" or key in (13, 10):
                with db.tx():
                    db.update(row["image_id"], review="keep", review_reason=None, reviewer=reviewer,
                              reviewed_at=now, status="kept", reason=None)
                tally["keep"] += 1
                i += 1
            elif ch == "r":
                pending = True
            elif ch == "b":
                i = max(0, i - 1)
                prev = db.get(rows[i]["image_id"])
                if prev["review"] == "reject":     # allow changing a rejection made in this session
                    with db.tx():
                        db.update(prev["image_id"], status="composed", reason=None, review=None)
            elif ch == "s":
                i += 1
        cv2.destroyAllWindows()
    progress(f"review: {tally}  (run again to continue with the rest)")
    return tally

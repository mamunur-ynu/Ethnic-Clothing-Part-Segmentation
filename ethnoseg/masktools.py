"""Tools to check and correct the part masks.

  python -m ethnoseg mask-sheets      contact sheets, one per group x class, weakest first
                                      -> reports/private/sheets/<GROUP>_<class>.jpg   (contain photos: do not publish)
  python -m ethnoseg mask-check       rule check + list of weak masks -> reports/weak_masks.csv
  python -m ethnoseg mask-edit [IDs]  small paint editor; with no IDs it opens the images on
                                      reports/fix_list.csv (from review) and reports/weak_masks.csv
  python -m ethnoseg mask-agreement   SAM 3 mask vs. corrected mask (IoU per class)
                                      -> reports/mask_agreement.md / .json

Editor keys:  0-6 choose class (0 = background, i.e. erase)   left mouse = paint
              [ / ]  smaller / bigger brush     U undo     S save     N next     Q quit
Before the FIRST save of an image, the original SAM 3 mask is copied to work/mask_auto/<ID>.png,
so the automatic and the corrected mask can be compared later (mask-agreement).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from . import imgops
from .manifest import Manifest
from .review import _font
from .settings import Settings

TINY_SHARE = 0.003        # a class covering < 0.3% of the subject is "tiny"
MIXED_SHARE = 0.10        # dress AND top/bottom each > 10% of the clothing -> "mixed"


def _final_ids(S: Settings) -> list:
    with Manifest(S.db_path) as db:
        return [(r["image_id"], r["grp"]) for r in db.rows("status='kept'")]


def _load(S: Settings, iid: str):
    img = np.asarray(Image.open(S.work_dir / "img" / f"{iid}.png").convert("RGB"))
    lab = np.asarray(Image.open(S.work_dir / "mask" / f"{iid}.png")).copy()
    sp = S.work_dir / "subject" / f"{iid}.png"
    sub = np.asarray(Image.open(sp)) > 127 if sp.exists() else lab > 0
    return img, lab, sub


# --------------------------------------------------------------------------- rule check
def check_one(lab: np.ndarray, sub: np.ndarray, part_ids: list[int]) -> dict:
    """Rule check of one mask. Returns problem flags and the class shares."""
    flags = []
    ids = set(np.unique(lab).tolist())
    if not ids <= {0, *part_ids}:
        flags.append("bad_class_id")
    if (lab[~sub] > 0).any():
        flags.append("label_outside_subject")
    area = max(1, int(sub.sum()))
    share = {c: float((lab == c).sum()) / area for c in part_ids}
    tiny = [c for c, s in share.items() if 0 < s < TINY_SHARE]
    if tiny:
        flags.append("tiny_region:" + "/".join(map(str, tiny)))
    clothing = sum(share.values()) or 1
    dress = share.get(3, 0) / clothing
    if dress > MIXED_SHARE and (share.get(1, 0) + share.get(2, 0)) / clothing > MIXED_SHARE:
        flags.append("dress_mixed_with_top_bottom")
    if clothing < 0.15:
        flags.append("little_clothing_labelled")
    return {"flags": flags, "share": share}


def check(S: Settings, progress=print) -> dict:
    part_ids = [c["id"] for c in S.part_classes]
    rows, hard = [], 0
    for iid, g in _final_ids(S):
        img, lab, sub = _load(S, iid)
        r = check_one(lab, sub, part_ids)
        if r["flags"]:
            rows.append((iid, g, ";".join(r["flags"])))
            hard += any(f in ("bad_class_id", "label_outside_subject") for f in r["flags"])
    S.reports_dir.mkdir(parents=True, exist_ok=True)
    with open(S.reports_dir / "weak_masks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "problems"])
        w.writerows(sorted(rows, key=lambda r: (-r[2].count(";"), r[0])))
    progress(f"mask-check: {len(rows)} masks with a warning, {hard} breaking a hard rule "
             f"(class id or label outside subject) -> reports/weak_masks.csv")
    return {"warnings": len(rows), "hard": hard}


# --------------------------------------------------------------------------- contact sheets
def sheets(S: Settings, per_sheet: int = 24, tile: int = 160, progress=print) -> int:
    out = S.reports_dir / "private" / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    font = _font(12)
    items = defaultdict(list)                         # (group, class) -> [(share, iid)]
    for iid, g in _final_ids(S):
        lab = np.asarray(Image.open(S.work_dir / "mask" / f"{iid}.png"))
        area = max(1, int((lab > 0).sum()))
        for c in S.part_classes:
            n = int((lab == c["id"]).sum())
            if n:
                items[(g, c["name"])].append((n / area, iid))
    made = 0
    for (g, cname), lst in sorted(items.items()):
        lst.sort()                                    # smallest region first = most likely weak
        lst = lst[:per_sheet]
        cid = next(c["id"] for c in S.part_classes if c["name"] == cname)
        cols = 6
        sheet = Image.new("RGB", (cols * tile, ((len(lst) + cols - 1) // cols) * (tile + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (share, iid) in enumerate(lst):
            img = np.asarray(Image.open(S.work_dir / "img" / f"{iid}.png").convert("RGB")).astype(np.float32)
            lab = np.asarray(Image.open(S.work_dir / "mask" / f"{iid}.png"))
            on = lab == cid
            vis = img * 0.45 + 255 * 0.55                       # fade everything ...
            vis[on] = img[on] * 0.4 + np.array(imgops.PALETTE[cid]) * 0.6   # ... except this class
            t = Image.fromarray(vis.astype(np.uint8)).resize((tile, tile))
            x, y = (k % cols) * tile, (k // cols) * (tile + 16)
            sheet.paste(t, (x, y + 16))
            d.text((x + 3, y + 1), f"{iid} {share:.1%}", fill=(0, 0, 0), font=font)
        sheet.save(out / f"{g}_{cname}.jpg", quality=85)
        made += 1
    progress(f"mask-sheets: {made} sheets in {out} (they show source photos: keep them private)")
    return made


# --------------------------------------------------------------------------- editor
class Editor:
    """Paint state for one image; kept separate from the window so it can be tested."""

    def __init__(self, lab: np.ndarray, sub: np.ndarray):
        self.lab, self.sub = lab, sub
        self.cls, self.brush = 1, 8
        self.history: list[np.ndarray] = []
        self.changed = False

    def paint(self, x: int, y: int, start: bool = False) -> None:
        if start:
            self.history.append(self.lab.copy())
        yy, xx = np.ogrid[:self.lab.shape[0], :self.lab.shape[1]]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= self.brush ** 2
        self.lab[disk & self.sub] = self.cls          # labels can only go on the subject
        self.changed = True

    def undo(self) -> None:
        if self.history:
            self.lab = self.history.pop()


def save_mask(S: Settings, iid: str, lab: np.ndarray) -> None:
    auto = S.work_dir / "mask_auto"
    auto.mkdir(parents=True, exist_ok=True)
    if not (auto / f"{iid}.png").exists():            # keep the SAM 3 original once
        shutil.copy2(S.work_dir / "mask" / f"{iid}.png", auto / f"{iid}.png")
    Image.fromarray(lab.astype(np.uint8)).save(S.work_dir / "mask" / f"{iid}.png", optimize=True)
    with Manifest(S.db_path) as db, db.tx():
        row = db.get(iid)
        flags = set(filter(None, (row["flags"] or "").split(",")))
        db.update(iid, flags=",".join(sorted(flags | {"mask_edited"})))


def queue_from_reports(S: Settings) -> list[str]:
    ids = []
    for name in ("fix_list.csv", "weak_masks.csv"):
        p = S.reports_dir / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                ids += [r["image_id"] for r in csv.DictReader(f)]
    kept = {i for i, _ in _final_ids(S)}
    return [i for i in dict.fromkeys(ids) if i in kept]


def edit(S: Settings, ids: list[str] | None = None, cv2=None, progress=print) -> int:
    if cv2 is None:
        import cv2  # noqa: F811
    ids = ids or queue_from_reports(S)
    if not ids:
        progress("mask-edit: nothing to edit (give image IDs, or run review-stats / mask-check first)")
        return 0
    win = "Mask editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    saved = 0
    for n, iid in enumerate(ids):
        img, lab, sub = _load(S, iid)
        ed = Editor(lab, sub)
        state = {"down": False}

        def on_mouse(event, x, y, flags, _param, ed=ed, state=state, w=img.shape[1]):
            x = x - w if x >= w else x                # paint on either half of the window
            if event == cv2.EVENT_LBUTTONDOWN:
                state["down"] = True
                ed.paint(x, y, start=True)
            elif event == cv2.EVENT_MOUSEMOVE and state["down"]:
                ed.paint(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["down"] = False

        cv2.setMouseCallback(win, on_mouse)
        while True:
            view = np.concatenate([img, imgops.overlay(img, ed.lab)], axis=1)
            cv2.putText(view, f"{n + 1}/{len(ids)} {iid}  class {ed.cls}  brush {ed.brush}  "
                              "0-6 class [ ] brush U undo S save N next Q quit",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.imshow(win, view[:, :, ::-1])
            key = cv2.waitKey(30) & 0xFF
            ch = chr(key).lower() if key < 128 else ""
            if ch.isdigit() and int(ch) <= 6:
                ed.cls = int(ch)
            elif ch == "[":
                ed.brush = max(1, ed.brush - 2)
            elif ch == "]":
                ed.brush = min(60, ed.brush + 2)
            elif ch == "u":
                ed.undo()
            elif ch == "s":
                save_mask(S, iid, ed.lab)
                saved += 1
                ed.changed = False
            elif ch in ("n", "q") or key == 27:
                break
        if ch == "q" or key == 27:
            break
    cv2.destroyAllWindows()
    progress(f"mask-edit: saved {saved} corrected masks (originals kept in work/mask_auto/). "
             "Run split/export/qc again to use them.")
    return saved


# --------------------------------------------------------------------------- agreement
def agreement(S: Settings, progress=print) -> dict:
    auto_dir = S.work_dir / "mask_auto"
    names = {c["id"]: c["name"] for c in S.part_classes}
    inter, union = defaultdict(int), defaultdict(int)
    same = total = n = 0
    per_image = []
    for p in sorted(auto_dir.glob("*.png")) if auto_dir.exists() else []:
        a = np.asarray(Image.open(p))
        h = np.asarray(Image.open(S.work_dir / "mask" / p.name))
        n += 1
        same += int((a == h).sum())
        total += a.size
        ious = []
        for c in names:
            i, u = int(((a == c) & (h == c)).sum()), int(((a == c) | (h == c)).sum())
            inter[c] += i
            union[c] += u
            if u:
                ious.append(i / u)
        per_image.append((p.stem, float(np.mean(ious)) if ious else 1.0))
    iou = {names[c]: round(inter[c] / union[c], 3) for c in names if union[c]}
    result = {"images_corrected": n, "pixel_agreement": round(same / total, 4) if total else None,
              "iou_per_class": iou, "mean_iou": round(float(np.mean(list(iou.values()))), 3) if iou else None,
              "lowest_images": sorted(per_image, key=lambda x: x[1])[:10]}
    S.reports_dir.mkdir(parents=True, exist_ok=True)
    (S.reports_dir / "mask_agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# SAM 3 mask vs. human-corrected mask", "",
             f"Images corrected by hand: {n}", f"Pixel agreement: {result['pixel_agreement']}",
             f"Mean IoU over classes: {result['mean_iou']}", "", "| Class | IoU |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in iou.items()]
    (S.reports_dir / "mask_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(f"mask-agreement: {n} corrected masks; pixel agreement {result['pixel_agreement']}; "
             f"mean IoU {result['mean_iou']}")
    return result

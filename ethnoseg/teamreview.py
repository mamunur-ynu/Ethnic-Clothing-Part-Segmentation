"""Two-person human review with an overlap, so agreement can be measured.

Three commands:

  python -m ethnoseg review-plan --reviewers Mamun,Imamul
      Makes the work plan (table "assignments" in the manifest):
        - every composed image that has no decision yet is given to ONE reviewer
          (halves, seeded), and OVERLAP (10%) of them to BOTH reviewers;
        - an AUDIT sample (10%) of the images that already have a decision
          (e.g. from an earlier reviewer) is given to BOTH reviewers as well.
      Decisions already in the manifest are copied into the "reviews" table first,
      under the reviewer name that is stored with them, so nothing is lost.

  python -m ethnoseg review --reviewer Mamun       (no --reviewer: the tool asks for the name)
      Opens the review window with ONLY this person's assigned images. Other people's
      decisions are NOT shown (a blind review, so one person does not copy the other).
      Keys:
          K / Enter   keep                       R then 1-7   reject with a reason
          F then 1-6  mark a mask problem (fix list; press again to remove)
          B back      S skip      Q / Esc quit (everything is saved at once)

  python -m ethnoseg review-stats
      Counts per reviewer, reject reasons, Cohen's kappa for each pair of reviewers,
      conflicts, and the fix list -> reports/review_stats.md / .json, reports/fix_list.csv

Final decision for each image = majority of all decisions on it. With two votes that
disagree, the image is rejected ("review_conflict") and listed, so a person can look again.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

from . import imgops
from .manifest import Manifest
from .review import REASONS, _font
from .settings import Settings

FIXES = {
    "1": "headwear_missing",       # hat / headdress not labelled
    "2": "top_bottom_dress_mixed",  # wrong garment class on a big area
    "3": "skin_or_hair_labelled",   # face, hands or hair painted as clothing
    "4": "tiny_or_broken_regions",
    "5": "background_in_mask",
    "6": "other_fix",
}

ASSIGN_SQL = """
CREATE TABLE IF NOT EXISTS assignments (
    image_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    kind     TEXT NOT NULL,              -- new / overlap / audit
    PRIMARY KEY (image_id, reviewer)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _order(seed: int, image_id: str) -> str:
    return hashlib.sha1(f"{seed}:{image_id}".encode()).hexdigest()


def has_plan(S: Settings) -> bool:
    with Manifest(S.db_path) as db:
        db.con.executescript(ASSIGN_SQL)
        return db.con.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] > 0


# --------------------------------------------------------------------------- plan
def plan(S: Settings, reviewers: list[str], overlap: float = 0.10, audit: float = 0.10,
         progress=print) -> dict:
    if len(reviewers) != 2:
        raise SystemExit("Give exactly two reviewer names, e.g. --reviewers Mamun,Imamul")
    a, b = reviewers
    with Manifest(S.db_path) as db, db.tx():
        db.con.executescript(ASSIGN_SQL)
        # step 1: keep earlier decisions as reviews by the person stored with them
        copied = 0
        for r in db.rows("review IS NOT NULL AND reviewer IS NOT NULL"):
            for who in r["reviewer"].split(","):
                if not db.con.execute("SELECT 1 FROM reviews WHERE image_id=? AND reviewer=?",
                                      (r["image_id"], who)).fetchone():
                    db.add_review(r["image_id"], who, r["review"], r["review_reason"], None,
                                  r["reviewed_at"] or _now())
                    copied += 1
        # step 2: images still to decide -> halves, plus an overlap for both
        new = sorted((r["image_id"] for r in db.rows("status='composed' AND review IS NULL")),
                     key=lambda i: _order(S.seed, i))
        n_over = round(len(new) * overlap)
        both, rest = new[:n_over], new[n_over:]
        rows = [(i, a, "overlap") for i in both] + [(i, b, "overlap") for i in both]
        rows += [(i, a if k % 2 == 0 else b, "new") for k, i in enumerate(rest)]
        # step 3: audit sample of images that already have a decision (still valid images only:
        # final 'kept', or rejected by a person - not those rejected by automatic rules)
        decided = [r["image_id"] for r in db.rows(
            "review IS NOT NULL AND (status='kept' OR (status='rejected' AND reason LIKE 'review:%'))")]
        decided.sort(key=lambda i: _order(S.seed + 1, i))
        aud = decided[:round(len(decided) * audit)]
        rows += [(i, a, "audit") for i in aud] + [(i, b, "audit") for i in aud]
        db.con.executemany("INSERT OR IGNORE INTO assignments(image_id, reviewer, kind) VALUES (?,?,?)", rows)
        db.log("review-plan", f"{a}/{b}: new={len(rest)} overlap={len(both)} audit={len(aud)} copied={copied}")
        per = Counter(r[1] for r in rows)
    progress(f"review-plan: copied {copied} earlier decisions; new {len(rest)} (split in two), "
             f"overlap {len(both)} and audit {len(aud)} (both reviewers). Images per person: {dict(per)}")
    return {"copied": copied, "new": len(rest), "overlap": len(both), "audit": len(aud), "per": dict(per)}


# --------------------------------------------------------------------------- final decision
def resolve(db: Manifest, image_id: str) -> None:
    """Set the final decision of one image from all its reviews (majority vote)."""
    votes = db.reviews("image_id=?", (image_id,))
    if not votes:
        return
    row = db.get(image_id)
    if row["status"] not in ("composed", "kept") and not (row["status"] == "rejected" and
                                                         (row["reason"] or "").startswith("review")):
        return                                   # rejected by an automatic rule: leave it
    keep = sum(v["decision"] == "keep" for v in votes)
    rej = len(votes) - keep
    who = ",".join(v["reviewer"] for v in votes)
    last = max(v["ts"] or "" for v in votes)
    if keep > rej:
        db.update(image_id, review="keep", review_reason=None, reviewer=who, reviewed_at=last,
                  status="kept", reason=None)
    else:
        reasons = Counter(v["reason"] for v in votes if v["decision"] == "reject" and v["reason"])
        why = "review_conflict" if keep == rej else (reasons.most_common(1)[0][0] if reasons else "other")
        db.update(image_id, review="reject", review_reason=why, reviewer=who, reviewed_at=last,
                  status="rejected", reason=f"review:{why}")


# --------------------------------------------------------------------------- window
def render_blind(S: Settings, row, index: int, total: int, mode: str, fixes: set[str]) -> np.ndarray:
    """Review canvas without anyone's earlier decision on it."""
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
    ev = (row["evidence"] or "")[:88]
    for k in range(0, len(ev), 22):
        d.text((12, y), ev[k:k + 22], fill=(60, 60, 60), font=fs); y += 20
    y += 8
    for c in S.part_classes:
        d.rectangle((12, y + 3, 28, y + 17), fill=tuple(int(v) for v in imgops.PALETTE[c["id"]]))
        d.text((36, y), f"{c['id']} {c['name']}", fill=(0, 0, 0), font=fs); y += 21
    y += 8
    if fixes:
        d.text((12, y), "Mask problems: " + ", ".join(sorted(fixes))[:40], fill=(200, 90, 0), font=fs); y += 24
    if mode == "reject":
        d.text((12, y), "Reject reason:", fill=(200, 0, 0), font=f); y += 26
        for k, v in REASONS.items():
            d.text((12, y), f"{k}  {v}", fill=(200, 0, 0), font=fs); y += 20
    elif mode == "fix":
        d.text((12, y), "Mask problem (toggle):", fill=(200, 90, 0), font=f); y += 26
        for k, v in FIXES.items():
            d.text((12, y), f"{k}  {v}", fill=(200, 90, 0), font=fs); y += 20
    else:
        for line in ("K/Enter keep   R reject", "F mark a mask problem", "B back  S skip  Q quit"):
            d.text((12, y), line, fill=(0, 0, 0), font=fs); y += 22
    return np.concatenate([left, np.asarray(panel)], axis=1)


def session(S: Settings, reviewer: str, cv2=None, progress=print, min_view: float = 0.0,
            clock=time.monotonic) -> dict:
    """One person's review session over their own assigned images."""
    if cv2 is None:
        import cv2  # noqa: F811
    tally = {"keep": 0, "reject": 0}
    with Manifest(S.db_path) as db:
        db.con.executescript(ASSIGN_SQL)
        todo = [r[0] for r in db.con.execute(
            "SELECT a.image_id FROM assignments a WHERE a.reviewer=? AND NOT EXISTS "
            "(SELECT 1 FROM reviews v WHERE v.image_id=a.image_id AND v.reviewer=a.reviewer) "
            "ORDER BY a.image_id", (reviewer,))]
        if not todo:
            progress(f"Nothing left for {reviewer}. (Run review-plan first if this is unexpected.)")
            return tally
        win = f"Review - {reviewer}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        i, mode = 0, ""
        fixes: dict[str, set[str]] = defaultdict(set)
        shown_i, shown_at, last_key_at = -1, 0.0, -1e9
        while 0 <= i < len(todo):
            row = db.get(todo[i])
            cv2.imshow(win, render_blind(S, row, i, len(todo), mode, fixes[todo[i]])[:, :, ::-1])
            if i != shown_i:
                shown_i, shown_at = i, clock()
            key = cv2.waitKey(0) & 0xFF
            ch = chr(key).lower() if key < 128 else ""
            t, prev, last_key_at = clock(), last_key_at, clock()
            deciding = mode == "" and (ch in "kr" or key in (13, 10)) or mode == "reject" and ch in REASONS
            if min_view > 0 and deciding and (t - shown_at < min_view or t - prev < 0.15):
                continue                                      # too soon, or a held-down key repeating
            if mode == "reject":
                mode = ""
                if ch in REASONS:
                    _decide(db, todo[i], reviewer, "reject", REASONS[ch], fixes[todo[i]])
                    tally["reject"] += 1
                    i += 1
                continue
            if mode == "fix":
                mode = ""
                if ch in FIXES:
                    fixes[todo[i]] ^= {FIXES[ch]}
                continue
            if ch == "q" or key == 27:
                break
            if ch == "k" or key in (13, 10):
                _decide(db, todo[i], reviewer, "keep", None, fixes[todo[i]])
                tally["keep"] += 1
                i += 1
            elif ch == "r":
                mode = "reject"
            elif ch == "f":
                mode = "fix"
            elif ch == "b" and i > 0:
                i -= 1
                with db.tx():                                 # undo this person's decision on it
                    db.con.execute("DELETE FROM reviews WHERE image_id=? AND reviewer=?", (todo[i], reviewer))
                    _reset_if_unreviewed(db, todo[i])
                    resolve(db, todo[i])
            elif ch == "s":
                i += 1
        cv2.destroyAllWindows()
    left = len(todo) - tally["keep"] - tally["reject"]
    progress(f"review ({reviewer}): {tally}; about {left} still to do (run again to continue)")
    return tally


def _decide(db: Manifest, image_id: str, reviewer: str, decision: str, reason, fixes: set[str]) -> None:
    with db.tx():
        db.add_review(image_id, reviewer, decision, reason, ",".join(sorted(fixes)) or None, _now())
        resolve(db, image_id)


def _reset_if_unreviewed(db: Manifest, image_id: str) -> None:
    """After 'back' removed the only decision on an image, make it undecided again."""
    if db.reviews("image_id=?", (image_id,)):
        return
    row = db.get(image_id)
    if row["status"] == "kept" or (row["status"] == "rejected" and (row["reason"] or "").startswith("review")):
        db.update(image_id, review=None, review_reason=None, reviewer=None, reviewed_at=None,
                  status="composed", reason=None)


# --------------------------------------------------------------------------- statistics
def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two lists of keep/reject decisions on the same images."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def stats(S: Settings, progress=print) -> dict:
    out = S.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    with Manifest(S.db_path) as db:
        db.con.executescript(ASSIGN_SQL)
        reviews = db.reviews()
        assigned = Counter(r[0] for r in db.con.execute("SELECT reviewer FROM assignments"))
        groups = {r["image_id"]: r["grp"] for r in db.rows()}
    by_img: dict[str, dict[str, dict]] = defaultdict(dict)
    for v in reviews:
        by_img[v["image_id"]][v["reviewer"]] = dict(v)
    people = sorted({v["reviewer"] for v in reviews})
    per = {p: Counter(v["decision"] for v in reviews if v["reviewer"] == p) for p in people}
    reasons = {p: Counter(v["reason"] for v in reviews if v["reviewer"] == p and v["reason"]) for p in people}
    pairs = []
    for p, q in combinations(people, 2):
        common = [i for i, d in by_img.items() if p in d and q in d]
        if not common:
            continue
        a = [by_img[i][p]["decision"] for i in common]
        b = [by_img[i][q]["decision"] for i in common]
        agree = sum(x == y for x, y in zip(a, b))
        k = cohen_kappa(a, b)
        pairs.append({"pair": f"{p} vs {q}", "images": len(common), "agreement": round(agree / len(common), 3),
                      "kappa": None if k is None else round(k, 3)})
    conflicts = [i for i, d in by_img.items() if len(d) >= 2 and
                 sum(x["decision"] == "keep" for x in d.values()) * 2 == len(d)]
    fixes = [(i, p, v["fix"]) for i, d in by_img.items() for p, v in d.items() if v.get("fix")]
    with open(out / "fix_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "reviewer", "problems"])
        w.writerows([(i, groups.get(i, ""), p, fx) for i, p, fx in sorted(fixes)])
    result = {"reviewers": people, "assigned": dict(assigned),
              "decisions": {p: dict(c) for p, c in per.items()},
              "done_of_assigned": {p: f"{sum(per.get(p, Counter()).values())}/{assigned.get(p, 0)}"
                                   for p in assigned},
              "reject_reasons": {p: dict(c) for p, c in reasons.items()},
              "pairs": pairs, "conflicts": sorted(conflicts), "fix_list_rows": len(fixes)}
    (out / "review_stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Review statistics", "", "| Reviewer | keep | reject | assigned done |", "|---|---|---|---|"]
    for p in people:
        lines.append(f"| {p} | {per[p].get('keep', 0)} | {per[p].get('reject', 0)} | "
                     f"{result['done_of_assigned'].get(p, '-')} |")
    lines += ["", "| Pair | images in common | agreement | Cohen's kappa |", "|---|---|---|---|"]
    lines += [f"| {x['pair']} | {x['images']} | {x['agreement']:.1%} | {x['kappa']} |" for x in pairs]
    lines += ["", f"Conflicts (tie, rejected until someone looks again): {len(conflicts)}",
              f"Mask problems marked (fix list rows): {len(fixes)} -> reports/fix_list.csv"]
    (out / "review_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("review-stats: " + "; ".join(f"{x['pair']} kappa={x['kappa']} on {x['images']}" for x in pairs)
             + f"; conflicts {len(conflicts)}; fix list {len(fixes)}")
    return result

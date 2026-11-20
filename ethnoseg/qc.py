"""Stage 8: check dataset/ against the brief and write dataset/qc_report.md (+ .json).

Hard failures (exit code 1) break a rule of the brief; warnings are worth a look.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from . import imgops
from .export import META_FIELDS
from .settings import Settings
from .web import strict_evidence

REQUIRED = ["image_id", "path", "ethnicity", "garment_type", "split", "source_url",
            "publisher", "collected_at", "license", "processing"]


def run(S: Settings, progress=print) -> dict:
    d, q = S.dataset_dir, S.section("qc")
    hard: dict[str, list] = defaultdict(list)
    warn: dict[str, list] = defaultdict(list)
    meta_path = d / "metadata.csv"
    if not meta_path.exists():
        raise SystemExit("dataset/metadata.csv not found: run export first")
    with open(meta_path, encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))
    classes = json.loads((d / "annotations" / "classes.json").read_text(encoding="utf-8"))
    valid_ids = {int(k) for k in classes}
    expected = {str(c["id"]): c["name"] for c in S.classes}
    if classes != expected:
        hard["classes_json_mismatch"].append(str(classes))

    missing_cols = [c for c in META_FIELDS if meta and c not in meta[0]]
    if missing_cols:
        hard["metadata_missing_columns"] += missing_cols
    groups = {g["code"]: g for g in S.raw["groups"]}
    ids = Counter(m["image_id"] for m in meta)
    hard["duplicate_image_id"] += [k for k, v in ids.items() if v > 1]

    size = S.section("compose")["size"]
    pixel = Counter()
    images_present = 0
    hashes, thumbs = {}, {}
    for m in meta:
        iid = m["image_id"]
        for col in REQUIRED:
            if not (m.get(col) or "").strip():
                hard[f"empty_{col}"].append(iid)
        if not m["verify_status"].startswith("verified"):
            hard["unverified_label"].append(iid)
        g = groups.get(m["ethnicity_code"])
        if g is not None and S.section("collect").get("strict_evidence", True):
            ok, _ = strict_evidence(g, m.get("page_title", ""), m.get("evidence", ""))
            if not ok:
                hard["evidence_not_one_group"].append(iid)
        if m["split"] not in S.section("split")["ratios"]:
            hard["bad_split_value"].append(iid)
        if m["review"] != "keep":
            warn["not_human_reviewed"].append(iid)
        mp = d / m["mask_path"]
        if not mp.exists():
            hard["mask_missing"].append(iid)
            continue
        mask = np.asarray(Image.open(mp))
        if mask.ndim != 2 or mask.dtype != np.uint8:
            hard["mask_not_single_channel_uint8"].append(iid)
        if mask.shape != (size, size):
            hard["mask_wrong_size"].append(iid)
        bad = set(np.unique(mask).tolist()) - valid_ids
        if bad:
            hard["mask_bad_class_id"].append(f"{iid}:{sorted(bad)}")
        pixel.update(dict(zip(*np.unique(mask, return_counts=True))))
        if mp.stem != iid or (d / m["path"]).stem != iid:
            hard["name_mismatch"].append(iid)
        ip = d / m["path"]
        if not ip.exists():
            continue                                  # --no-images build
        images_present += 1
        img = Image.open(ip)
        if img.mode != "RGB" or img.size != (size, size):
            hard["image_wrong_mode_or_size"].append(iid)
            continue
        arr = np.asarray(img)
        
        sp = S.work_dir / "subject" / f"{iid}.png"
        subj = np.asarray(Image.open(sp)) > 127 if sp.exists() else (mask > 0)
        hashes[iid] = imgops.subject_hash(arr, subj)
        thumbs[iid] = imgops.subject_thumb(arr, subj)
        if not sp.exists():
            warn["subject_mask_missing_used_labels"].append(iid)
        if not (arr[~subj] == 255).all():
            hard["background_not_pure_white"].append(iid)
        if (mask[~subj] != 0).any():
            hard["labels_outside_subject"].append(iid)
        dx, dy, fill = imgops.placement(subj)
        if dx > q["max_center_offset"] or dy > q["max_center_offset"]:
            hard["off_centre"].append(f"{iid}:{dx:.3f},{dy:.3f}")
        if fill < q["min_fill"]:
            hard["subject_too_small"].append(f"{iid}:{fill:.2f}")
        if (mask > 0).sum() < 0.05 * subj.sum():
            warn["very_little_clothing_labelled"].append(iid)

    # splits
    by_group = defaultdict(Counter)
    for m in meta:
        by_group[m["ethnicity_code"]][m["split"]] += 1
    ratios = S.section("split")["ratios"]
    for code, c in by_group.items():
        n = sum(c.values())
        for s, r in ratios.items():
            if abs(c[s] / n - r) > q["split_tolerance"]:
                warn["split_ratio_off"].append(f"{code}/{s}: {c[s] / n:.2f} vs {r}")
        if n < q["min_per_group"]:
            hard["group_below_minimum"].append(f"{code}: {n} < {q['min_per_group']}")
    if len(meta) < q["min_total"]:
        hard["total_below_minimum"].append(f"{len(meta)} < {q['min_total']}")
    if len(by_group) < 5:
        hard["fewer_than_5_groups"].append(str(len(by_group)))
    sizes = [sum(c.values()) for c in by_group.values()]
    balance = (max(sizes) / min(sizes)) if sizes else 0

    page_split = defaultdict(set)
    for m in meta:
        page_split[m["source_url"]].add(m["split"])
    hard["source_page_in_two_splits"] += [u for u, s in page_split.items() if len(s) > 1]
    split_of = {m["image_id"]: m["split"] for m in meta}
    keys = list(hashes)
    cc = S.section("compose")
    for i, j, _ in imgops.hamming_pairs([hashes[k] for k in keys], cc["final_near_dup_hamming"]):
        a, b = keys[i], keys[j]
        if imgops.thumb_diff(thumbs[a], thumbs[b]) > cc["final_pixel_diff"]:
            continue
        if split_of[a] != split_of[b]:
            hard["near_duplicate_across_splits"].append(f"{a}~{b}")
        else:
            warn["near_duplicate_same_split"].append(f"{a}~{b}")
    for c in valid_ids - {0}:
        if pixel.get(c, 0) == 0:
            warn["class_never_used"].append(classes[str(c)])

    allpx = max(1, sum(pixel.values()))
    result = {
        "images": len(meta), "images_checked_on_disk": images_present,
        "per_group": {k: dict(v) for k, v in by_group.items()},
        "largest_to_smallest_group": round(balance, 2),
        "pixel_share": {classes.get(str(k), f"INVALID_{k}"): round(v / allpx, 4) for k, v in sorted(pixel.items())},
        "hard_failures": {k: len(v) for k, v in hard.items() if v},
        "warnings": {k: len(v) for k, v in warn.items() if v},
        "examples": {k: v[:10] for k, v in {**hard, **warn}.items() if v},
    }
    (d / "qc_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "qc_report.md").write_text(_markdown(result), encoding="utf-8")
    ok = not result["hard_failures"]
    progress(f"QC {'PASSED' if ok else 'FAILED'}: {len(meta)} images, "
             f"hard failures {result['hard_failures'] or 'none'}, warnings {result['warnings'] or 'none'}")
    return result


def _markdown(r: dict) -> str:
    lines = ["# QC report", "", f"- Images: **{r['images']}** (checked on disk: {r['images_checked_on_disk']})",
             f"- Largest / smallest group: {r['largest_to_smallest_group']}", "",
             "## Result", "", ("**PASSED** - no hard failures." if not r["hard_failures"] else "**FAILED**"), ""]
    for title, key in (("Hard failures", "hard_failures"), ("Warnings", "warnings")):
        lines += [f"### {title}", ""]
        if not r[key]:
            lines += ["None.", ""]
        for k, n in r[key].items():
            lines.append(f"- `{k}`: {n} (e.g. {', '.join(map(str, r['examples'].get(k, [])[:5]))})")
        lines.append("")
    lines += ["## Per group", "", "| Group | " + " | ".join(["train", "val", "test"]) + " |", "|---|---|---|---|"]
    for g, c in r["per_group"].items():
        lines.append(f"| {g} | {c.get('train', 0)} | {c.get('val', 0)} | {c.get('test', 0)} |")
    lines += ["", "## Pixel share by class", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in r["pixel_share"].items()]
    lines += ["", "## Checks performed", "",
              "classes.json matches config; metadata columns and required fields; unique ids; "
              "verified source labels; mask is single-channel uint8, 512x512, valid ids, same name as image; "
              "image RGB 512x512; background pixels exactly (255,255,255); labels only on the subject; "
              "subject centre within 10% and longer side >= 60%; per-group and total minimums; split ratios; "
              "no source page or near-duplicate across splits; every class used.", ""]
    return "\n".join(lines)

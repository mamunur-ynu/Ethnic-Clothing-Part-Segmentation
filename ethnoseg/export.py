"""Stage 7: write dataset/ in the layout required by the brief.

dataset/
  images/{train,val,test}/<ID>.png
  annotations/masks/{train,val,test}/<ID>.png     same name and size as the image
  annotations/classes.json                         id -> class name
  annotations/ethnicity_map.json                   code -> English / Chinese name
  annotations/classification_labels.csv            image_id,path,ethnicity,garment_type,split
  metadata.csv                                     one row per image, source + processing fields
  README.md                                        generated: categories, rules, statistics, steps
"""
from __future__ import annotations

import csv
import json
import platform
import shutil
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

from . import __version__
from .manifest import Manifest
from .settings import Settings

META_FIELDS = ["image_id", "path", "mask_path", "ethnicity", "ethnicity_zh", "ethnicity_code",
               "garment_type", "split", "source_url", "image_url", "publisher", "page_title",
               "collected_at", "license", "verify_status", "evidence_type", "evidence", "source_category",
               "search_query", "processing", "review", "reviewer", "reviewed_at", "cluster"]


def processing_string(S: Settings) -> str:
    seg, comp = S.section("segment"), S.section("compose")
    return (f"sam3({seg['model_id']}) subject+parts @{seg['work_max_side']}px, "
            f"part_thr={seg['part_threshold']}; white bg; fit={comp['fill']}; pad {comp['size']}x{comp['size']}; "
            f"human review")


def run(S: Settings, include_unreviewed: bool = False, no_images: bool = False, progress=print) -> dict:
    out = S.dataset_dir
    if out.exists():
        shutil.rmtree(out)
    statuses = ("kept", "composed") if include_unreviewed else ("kept",)
    codes = [g["code"] for g in S.groups()]
    with Manifest(S.db_path) as db:
        rows = [r for r in db.rows(f"status IN ({','.join('?' * len(statuses))}) AND split IS NOT NULL", statuses)
                if r["grp"] in codes]
    if not rows:
        raise SystemExit("No images to export: run review and split first "
                         "(or pass --include-unreviewed for a provisional build).")
    proc = processing_string(S)
    meta, labels = [], []
    pixel = Counter()
    for r in rows:
        g = S.group(r["grp"])
        sp = r["split"]
        img_rel = f"images/{sp}/{r['image_id']}.png"
        mask_rel = f"annotations/masks/{sp}/{r['image_id']}.png"
        (out / img_rel).parent.mkdir(parents=True, exist_ok=True)
        (out / mask_rel).parent.mkdir(parents=True, exist_ok=True)
        if not no_images:
            shutil.copy2(S.work_dir / "img" / f"{r['image_id']}.png", out / img_rel)
        shutil.copy2(S.work_dir / "mask" / f"{r['image_id']}.png", out / mask_rel)
        m = np.asarray(Image.open(out / mask_rel))
        pixel.update(dict(zip(*np.unique(m, return_counts=True))))
        meta.append({
            "image_id": r["image_id"], "path": img_rel, "mask_path": mask_rel,
            "ethnicity": g["name"], "ethnicity_zh": g["zh"], "ethnicity_code": g["code"],
            "garment_type": r["garment_type"], "split": sp, "source_url": r["page_url"],
            "image_url": r["image_url"], "publisher": r["publisher"], "page_title": r["page_title"],
            "collected_at": r["collected_at"], "license": r["license"], "verify_status": r["verify_status"],
            "evidence_type": r["evidence_type"] or "", "evidence": r["evidence"],
            "source_category": r["source_category"] or "",
            "search_query": r["search_query"], "processing": proc,
            "review": r["review"] or "pending", "reviewer": r["reviewer"] or "", "reviewed_at": r["reviewed_at"] or "",
            "cluster": r["cluster"],
        })
        labels.append({"image_id": r["image_id"], "path": img_rel, "ethnicity": g["name"],
                       "garment_type": r["garment_type"], "split": sp})

    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    (ann / "classes.json").write_text(json.dumps({str(c["id"]): c["name"] for c in S.classes}, indent=2),
                                      encoding="utf-8")
    (ann / "ethnicity_map.json").write_text(json.dumps(
        {g["code"]: {"name": g["name"], "zh": g["zh"]} for g in S.groups()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for path, fields, data in ((out / "metadata.csv", META_FIELDS, meta),
                               (ann / "classification_labels.csv",
                                ["image_id", "path", "ethnicity", "garment_type", "split"], labels)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel shows Chinese text
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (out / "README.md").write_text(readme(S, meta, pixel, include_unreviewed, no_images), encoding="utf-8")
    progress(f"export: {len(meta)} images -> {out}" + ("  (PROVISIONAL: includes unreviewed)" if include_unreviewed else ""))
    return {"images": len(meta)}


def _team_lines(S: Settings) -> list[str]:
    """Team names from docs/report_info.yaml (if the file exists)."""
    info = S.root / "docs" / "report_info.yaml"
    if not info.exists():
        return []
    import yaml
    members = (yaml.safe_load(info.read_text(encoding="utf-8")) or {}).get("members", [])
    if not members:
        return []
    return ["", "## Team", ""] + [f"- {m['name']} ({m['id']})" for m in members]


def readme(S: Settings, meta: list[dict], pixel: Counter, provisional: bool, no_images: bool) -> str:
    splits = list(S.section("split")["ratios"])
    by = Counter((m["ethnicity_code"], m["split"]) for m in meta)
    lines = [f"# {S.raw['project']['name']}", ""]
    if provisional:
        lines += ["> **Provisional build:** includes images that have not been human-reviewed yet.", ""]
    lines += [
        f"Semantic segmentation dataset of Chinese ethnic clothing: {len(meta)} images, "
        f"{len(S.groups())} ethnic groups, {len(S.part_classes)} clothing-part classes. "
        f"Built {datetime.now():%Y-%m-%d} with ethnoseg {__version__}.", "",
        "## Ethnic categories", "", "| Code | Ethnicity | 民族 | " + " | ".join(splits) + " | Total |",
        "|---|---|---|" + "---|" * (len(splits) + 1)]
    for g in S.groups():
        n = [by[(g["code"], s)] for s in splits]
        lines.append(f"| {g['code']} | {g['name']} | {g['zh']} | " + " | ".join(map(str, n)) + f" | {sum(n)} |")
    tot = [sum(by[(g['code'], s)] for g in S.groups()) for s in splits]
    lines += [f"| | **Total** | | " + " | ".join(map(str, tot)) + f" | **{sum(tot)}** |", "",
              "Labels come only from the source page (see `verify_status` and `evidence` in metadata.csv); "
              "they were never inferred from how the clothing or the person looks.", "",
              "## Mask classes (`annotations/classes.json`)", "", "| ID | Class | SAM 3 prompts | Pixel share |",
              "|---|---|---|---|"]
    allpx = max(1, sum(pixel.values()))
    for c in S.classes:
        prompts = ", ".join(c.get("prompts", [])) or "-"
        lines.append(f"| {c['id']} | {c['name']} | {prompts} | {100 * pixel.get(c['id'], 0) / allpx:.1f}% |")
    seg = S.section("segment")
    lines += ["", "## Annotation rules", "",
              "- 0 = background **and** every non-clothing pixel of the subject (skin, hair, hands).",
              "- Only visible pixels are labelled; hidden parts are not guessed.",
              f"- Overlaps: classes are painted in the order {', '.join(S.raw['paint_order'])}; a later class "
              "wins, so headwear/footwear/waistwear are drawn on top of the garment they sit on.",
              "- `top` = upper garment ending around the waist/hip; `bottom` = skirt or trousers; "
              "`dress` = one-piece garment or robe covering upper and lower body.",
              f"- Two-piece rule: when both top and bottom cover >= {seg['two_piece_min_share']:.0%} of the subject, "
              "remaining `dress` pixels are reassigned to top or bottom by vertical position.",
              "- `waistwear` = apron, belt or sash worn over another garment.",
              "- Each image has a same-name mask of identical size (512x512); masks are single-channel PNG.",
              "", "## Directory and fields", "",
              "- `images/<split>/<ID>.png`: RGB, pure white background (255,255,255), subject centred, "
              f"longer side = {S.section('compose')['fill']:.0%} of 512. PNG avoids JPEG noise in the white background.",
              "- `annotations/masks/<split>/<ID>.png`: class ids 0-6.",
              "- `annotations/classification_labels.csv`: image_id, path, ethnicity, garment_type, split.",
              "- `metadata.csv`: " + ", ".join(META_FIELDS) + ".",
              "- `garment_type`: parts present (>= 1% of clothing pixels), joined with `+`.",
              "- `cluster`: images from the same source page; a cluster is never split across train/val/test.",
              "", "## Processing", "", f"`{processing_string(S)}`", "",
              f"- Split: {', '.join(f'{s} {r:.0%}' for s, r in S.section('split')['ratios'].items())} per group, "
              f"seed {S.seed}.",
              "", "## Licence", "",
              "Images come from public web pages and are assumed to be all-rights-reserved. "
              + ("This package contains masks and metadata only; " if no_images else "")
              + "Use `source_url` / `image_url` in metadata.csv to obtain the originals; do not redistribute.",
              "", "## Software", "",
              f"- Python {platform.python_version()} on {platform.system()} (local stages)",
              f"- SAM 3: code https://github.com/facebookresearch/sam3, weights `{seg['model_id']}` on Hugging Face "
              "(gated), paper arXiv:2511.16719. Used through Hugging Face transformers (Sam3Model / Sam3Processor) "
              "on a Modal GPU.",
              f"- SAM 3 settings: person prompt \"{seg['person_prompt']}\" (score >= {seg['person_threshold']}), "
              f"part score >= {seg['part_threshold']}, mask threshold {seg['mask_threshold']}, images resized to "
              f"{seg['work_max_side']} px before SAM 3. All values are in config.yaml.",
              *_team_lines(S),
              "", "## How to rebuild", "", "See the project README (run steps 1-11).", ""]
    return "\n".join(lines)

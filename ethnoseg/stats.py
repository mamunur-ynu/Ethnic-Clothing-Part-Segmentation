"""Stage 9: tables and figures for the report and slides (reports/)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from . import imgops
from .manifest import Manifest
from .settings import Settings


def collection_funnel(S: Settings) -> dict:
    """Counts per group from the collector logs: candidates, verify outcomes, saved."""
    out = {}
    for g in S.groups():
        log = S.raw_dir / g["code"] / "log.jsonl"
        if not log.exists():
            continue
        last = {}
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["image_url"] not in last or r.get("image_id"):    # retries append; keep the outcome
                last[r["image_url"]] = r
        c = Counter(candidates=len(last))
        for r in last.values():
            c["saved" if r.get("image_id") else (r.get("reason") or "other")] += 1
        out[g["code"]] = dict(c)
    return out


def run(S: Settings, progress=print) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = S.reports_dir
    (rd / "figures").mkdir(parents=True, exist_ok=True)
    codes = [g["code"] for g in S.groups()]
    names = {g["code"]: f"{g['name']}\n{g['zh']}" for g in S.groups()}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    with Manifest(S.db_path) as db:
        rows = [r for r in db.rows() if r["grp"] in codes]
    funnel = collection_funnel(S)
    status = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for r in rows:
        status[r["grp"]][r["status"]] += 1
        if r["status"] == "rejected":
            reasons[r["grp"]][(r["reason"] or "other").split(":")[0]] += 1

    summary = {"collection": funnel, "status": {k: dict(v) for k, v in status.items()},
               "rejections": {k: dict(v) for k, v in reasons.items()}}

    meta_path = S.dataset_dir / "metadata.csv"
    meta = list(csv.DictReader(open(meta_path, encoding="utf-8-sig"))) if meta_path.exists() else []
    if meta:
        splits = list(S.section("split")["ratios"])
        cnt = Counter((m["ethnicity_code"], m["split"]) for m in meta)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bottom = np.zeros(len(codes))
        for s, col in zip(splits, ["#3a6ea5", "#f2a541", "#7fb069"]):
            vals = np.array([cnt[(c, s)] for c in codes])
            ax.bar([names[c] for c in codes], vals, bottom=bottom, label=s, color=col)
            bottom += vals
        for x, v in enumerate(bottom):
            ax.text(x, v + 2, str(int(v)), ha="center", fontsize=9)
        ax.set_ylabel("images")
        ax.set_title("Images per ethnic group and split")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(rd / "figures" / "images_per_group.png", dpi=200)
        plt.close(fig)

        # class presence and pixel share per group
        parts = S.part_classes
        presence = defaultdict(Counter)
        share = defaultdict(Counter)
        for m in meta:
            mask = np.asarray(Image.open(S.dataset_dir / m["mask_path"]))
            u, n = np.unique(mask, return_counts=True)
            for k, v in zip(u.tolist(), n.tolist()):
                if k:
                    presence[m["ethnicity_code"]][k] += 1
                    share[m["ethnicity_code"]][k] += v
        fig, ax = plt.subplots(figsize=(7, 3.6))
        left = np.zeros(len(codes))
        for c in parts:
            tot = np.array([sum(share[g].values()) or 1 for g in codes])
            vals = np.array([share[g][c["id"]] for g in codes]) / tot
            ax.barh([names[g] for g in codes], vals, left=left, label=c["name"],
                    color=np.array(imgops.PALETTE[c["id"]]) / 255)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of clothing pixels")
        ax.set_title("Clothing-part composition per group")
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(rd / "figures" / "part_share_per_group.png", dpi=200)
        plt.close(fig)
        summary["class_presence"] = {g: {next(c["name"] for c in parts if c["id"] == k): v for k, v in presence[g].items()}
                                     for g in codes}

        # example grid: 2 per group (image | overlay)
        tiles = []
        for g in codes:
            ex = [m for m in meta if m["ethnicity_code"] == g][:2]
            for m in ex:
                im = np.asarray(Image.open(S.work_dir / "img" / f"{m['image_id']}.png").convert("RGB"))
                mk = np.asarray(Image.open(S.dataset_dir / m["mask_path"]))
                tiles.append(np.concatenate([im, imgops.overlay(im, mk)], axis=1))
        if tiles:
            cols = 2
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.concatenate([np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)])
            (rd / "private").mkdir(parents=True, exist_ok=True)      # shows source photos: not public
            Image.fromarray(grid).resize((grid.shape[1] // 2, grid.shape[0] // 2)).save(
                rd / "private" / "examples.jpg", quality=90)

    # rejection reasons (pipeline after download)
    if reasons:
        allr = Counter()
        for c in reasons.values():
            allr.update(c)
        top = allr.most_common(10)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#888")
        ax.set_xlabel("images removed")
        ax.set_title("Why images were removed after download")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(rd / "figures" / "rejection_reasons.png", dpi=200)
        plt.close(fig)

    # full funnel, from search hits to the final dataset
    screen_reasons = {"corrupt", "too_small", "bad_aspect", "blurry", "near_duplicate_of",
                      "conflicting_labels", "corrupt_or_not_image"}
    strict_reasons = {"multi_group_evidence", "own_group_missing"}
    rkind = lambda r: (r["reason"] or "").split(":")[0]
    passed_screen = [r for r in rows if not (r["status"] == "rejected" and rkind(r) in screen_reasons)]
    passed_strict = [r for r in passed_screen if not (r["status"] == "rejected" and rkind(r) in strict_reasons)]
    sam3 = [r for r in passed_strict if r["seg_json"]]
    composed = [r for r in sam3 if r["final_dhash"]]
    reviewed = [r for r in composed if r["review"]]
    final = [r for r in rows if r["status"] == "kept"]
    hits = sum(v.get("candidates", 0) for v in funnel.values())
    stages = [("search hits", hits), ("label verified, saved", len(rows)),
              ("passed screening", len(passed_screen)), ("strict one-group evidence", len(passed_strict)),
              ("segmented by SAM 3", len(sam3)), ("composed 512x512", len(composed)),
              ("reviewed by a person", len(reviewed)), ("final dataset", len(final))]
    summary["funnel"] = dict(stages)
    summary["not_segmented_reserve"] = len([r for r in passed_strict if not r["seg_json"]])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([k for k, _ in stages][::-1], [v for _, v in stages][::-1], color="#3a6ea5")
    for y, (_, v) in enumerate(stages[::-1]):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("images (log scale)")
    ax.set_title("From search hits to the final dataset")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rd / "figures" / "funnel.png", dpi=200)
    plt.close(fig)

    # source categories of the final images
    from .web import source_category
    cats = Counter(r["source_category"] or source_category(r["page_url"]) for r in final)
    summary["source_categories_final"] = dict(cats.most_common())
    if cats:
        fig, ax = plt.subplots(figsize=(7, 3.0))
        items = cats.most_common()
        ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#888")
        tot = sum(cats.values())
        for y, (_, v) in enumerate(items[::-1]):
            ax.text(v, y, f" {v} ({v / tot:.0%})", va="center", fontsize=8)
        ax.set_xlabel("final images")
        ax.set_title("Where the final images come from (source page type)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(rd / "figures" / "source_categories.png", dpi=200)
        plt.close(fig)

    if meta:
        summary["split_counts"] = {g: {s: cnt[(g, s)] for s in splits} for g in codes}
        cls_split = Counter()
        for m in meta:
            mask = np.asarray(Image.open(S.dataset_dir / m["mask_path"]))
            for k in np.unique(mask).tolist():
                if k:
                    cls_split[(k, m["split"])] += 1
        summary["images_with_class_per_split"] = {
            c["name"]: {s: cls_split[(c["id"], s)] for s in splits} for c in S.part_classes}
        tot_px = sum(sum(v.values()) for v in share.values()) or 1
        summary["class_pixel_share_of_clothing"] = {
            c["name"]: round(sum(share[g][c["id"]] for g in codes) / tot_px, 4) for c in S.part_classes}

    (rd / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"stats: figures and summary.json written to {rd}")
    return summary

"""Stage 6: stratified 70/15/15 split that never separates related images.

Images from the same source page form one cluster (a page often shows the same outfit
from several angles, and can show two groups). Clusters are global, so a page never
lands in two splits even when its images belong to different groups. Clusters are
shuffled with the project seed, biggest first, and each goes to the split that is
furthest below its target for the groups it contains.
"""
from __future__ import annotations

import random
from collections import Counter

from .manifest import Manifest
from .settings import Settings


def assign(clusters: dict[str, list[tuple[str, str]]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """clusters: key -> [(image_id, group)]. Returns image_id -> split."""
    rng = random.Random(seed)
    keys = sorted(clusters)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(clusters[k]))          # stable sort keeps the random order within a size
    group_total = Counter(g for v in clusters.values() for _, g in v)
    counts = {s: Counter() for s in ratios}
    out = {}
    for k in keys:
        per_group = Counter(g for _, g in clusters[k])

        def need(s):
            return sum(n * (ratios[s] * group_total[g] - counts[s][g]) for g, n in per_group.items())

        best = max(ratios, key=lambda s: (need(s), ratios[s]))
        for iid, _ in clusters[k]:
            out[iid] = best
        counts[best].update(per_group)
    return out


def run(S: Settings, include_unreviewed: bool = False, progress=print) -> dict:
    statuses = ("kept", "composed") if include_unreviewed else ("kept",)
    ratios = S.section("split")["ratios"]
    codes = [g["code"] for g in S.groups()]
    with Manifest(S.db_path) as db, db.tx():
        db.con.execute("UPDATE images SET split=NULL, cluster=NULL")
        rows = [r for r in db.rows(f"status IN ({','.join('?' * len(statuses))})", statuses) if r["grp"] in codes]
        clusters: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            clusters.setdefault(r["page_url"] or r["image_id"], []).append((r["image_id"], r["grp"]))
        names = {k: f"c{n:05d}" for n, k in enumerate(sorted(clusters), 1)}
        mapping = assign(clusters, ratios, S.seed)
        for k, members in clusters.items():
            for iid, _ in members:
                db.update(iid, split=mapping[iid], cluster=names[k])
        report = {g: {s: 0 for s in ratios} for g in codes}
        for r in rows:
            report[r["grp"]][mapping[r["image_id"]]] += 1
        db.log("split", str(report))
    for code, c in report.items():
        progress(f"split {code}: {c}")
    return report

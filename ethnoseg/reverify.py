"""Re-check the label evidence of every image already in the manifest with the strict rule.

Rule (same as web.strict_evidence): the page title + evidence text must name the image's
own group and no other of the 56 groups. Images that fail are rejected with the reason
"multi_group_evidence" or "own_group_missing". Their review decision is kept (not deleted),
so the history stays visible.

It also fills evidence_type and source_category for every image.
Run:  python -m ethnoseg reverify          (add --dry-run to only print the counts)
"""
from __future__ import annotations

from collections import Counter

from .manifest import Manifest
from .settings import Settings
from .web import EVIDENCE_TYPE, source_category, strict_evidence

# Only images that are still in the pipeline are checked.
ACTIVE = ("candidate", "screened", "segmented", "composed", "kept")


def run(S: Settings, dry_run: bool = False, progress=print) -> dict:
    groups = {g["code"]: g for g in S.raw["groups"]}
    failed: Counter = Counter()
    lost_final: Counter = Counter()
    with Manifest(S.db_path) as db, db.tx():
        for row in db.rows():
            # step 1: fill the two descriptive fields for every image
            fields = dict(evidence_type=EVIDENCE_TYPE.get(row["verify_status"] or "", ""),
                          source_category=source_category(row["page_url"]))
            # step 2: apply the strict rule to images that are still active
            if row["status"] in ACTIVE:
                ok, why = strict_evidence(groups[row["grp"]], row["page_title"], row["evidence"])
                if not ok:
                    failed[(row["grp"], why)] += 1
                    if row["status"] == "kept":
                        lost_final[row["grp"]] += 1
                    fields.update(status="rejected", reason=why)
            if not dry_run:
                db.update(row["image_id"], **fields)
        if not dry_run:
            db.log("reverify", f"strict evidence rule: {dict(failed)}")
        if dry_run:
            db.con.rollback()
    by_reason = Counter()
    for (g, why), n in failed.items():
        by_reason[why] += n
    progress(f"reverify{' (dry run)' if dry_run else ''}: rejected {sum(failed.values())} "
             f"{dict(by_reason)}; removed from the final set per group: {dict(lost_final)}")
    return {"failed": dict(failed), "lost_final": dict(lost_final)}

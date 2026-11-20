"""Command line: python -m ethnoseg <command> [options]   (run from the project folder)"""
from __future__ import annotations

import argparse
import sys

from . import settings, utf8_console


def main(argv=None) -> int:
    utf8_console()
    p = argparse.ArgumentParser(prog="python -m ethnoseg", description=__doc__)
    p.add_argument("--config", help="path to config.yaml (default: project config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pull = sub.add_parser("pull", help="copy raw/ or seg/ from the Modal Volume into data/")
    pull.add_argument("folder", choices=["raw", "seg"])
    sub.add_parser("ingest", help="load data/raw/*/log.jsonl into data/manifest.db")
    sub.add_parser("screen", help="corrupt/small/blurry/banner/near-duplicate filtering")
    c = sub.add_parser("compose", help="white background, centre, 512x512, final masks")
    c.add_argument("--redo", action="store_true", help="rebuild images that are already composed")
    r = sub.add_parser("review", help="keyboard review window (keep / reject)")
    r.add_argument("--reviewer", default=None,
                   help="your name, saved with each decision (if left out, the tool asks for it)")
    r.add_argument("--all", action="store_true", help="also show images already reviewed")
    r.add_argument("--group", help="only one group code, e.g. YI")
    for name, helptext in (("split", "70/15/15 split"), ("export", "write dataset/")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--include-unreviewed", action="store_true",
                       help="provisional build that also uses images not yet reviewed")
        if name == "export":
            s.add_argument("--no-images", action="store_true", help="masks + metadata only (licence-safe package)")
    rp = sub.add_parser("review-plan", help="share the review between two people (with overlap)")
    rp.add_argument("--reviewers", required=True, help="two names, e.g. Mamun,Imamul")
    rp.add_argument("--overlap", type=float, default=0.10, help="share of new images both review")
    rp.add_argument("--audit", type=float, default=0.10, help="share of decided images both re-check")
    sub.add_parser("review-stats", help="agreement (Cohen's kappa), counts, conflicts, fix list")
    sub.add_parser("mask-sheets", help="contact sheets per group x class (weakest first)")
    sub.add_parser("mask-check", help="rule check + list of weak masks")
    me = sub.add_parser("mask-edit", help="paint editor for masks (default: fix list + weak masks)")
    me.add_argument("ids", nargs="*", help="image IDs to edit")
    sub.add_parser("mask-agreement", help="SAM 3 mask vs corrected mask (IoU per class)")
    rv = sub.add_parser("reverify", help="re-check label evidence with the strict one-group rule")
    rv.add_argument("--dry-run", action="store_true", help="only print what would change")
    sub.add_parser("qc", help="check dataset/ and write qc_report.md")
    sub.add_parser("stats", help="figures and tables for the report")
    sub.add_parser("status", help="counts per group and status")
    a = p.parse_args(argv)
    S = settings.load(a.config)

    if a.cmd == "pull":
        from . import sync
        sync.pull(S.raw["modal"]["volume"], a.folder, S.data_dir)
        if a.folder == "seg":
            from .manifest import Manifest
            with Manifest(S.db_path) as db, db.tx():
                n = 0
                for row in db.rows("status='screened'"):
                    if (S.seg_dir / f"{row['image_id']}.json").exists():
                        db.update(row["image_id"], status="segmented")
                        n += 1
            print(f"{n} images marked 'segmented'")
    elif a.cmd == "ingest":
        from . import ingest
        ingest.run(S)
    elif a.cmd == "screen":
        from . import screen
        screen.run(S)
    elif a.cmd == "compose":
        from . import compose
        compose.run(S, redo=a.redo)
    elif a.cmd == "review":
        from . import review, teamreview
        name = (a.reviewer or input("Reviewer name (e.g. Mamun or Imamul): ")).strip()
        if not name:
            raise SystemExit("A reviewer name is needed.")
        if teamreview.has_plan(S):          # a two-person plan exists: blind review of own images
            teamreview.session(S, name, min_view=0.4)
        else:                               # no plan: one person reviews everything still open
            review.run(S, name, include_done=a.all, group=a.group, min_view=0.4)
    elif a.cmd in ("mask-sheets", "mask-check", "mask-edit", "mask-agreement"):
        from . import masktools
        if a.cmd == "mask-sheets":
            masktools.sheets(S)
        elif a.cmd == "mask-check":
            masktools.check(S)
        elif a.cmd == "mask-edit":
            masktools.edit(S, a.ids)
        else:
            masktools.agreement(S)
    elif a.cmd == "review-plan":
        from . import teamreview
        teamreview.plan(S, [x.strip() for x in a.reviewers.split(",") if x.strip()], a.overlap, a.audit)
    elif a.cmd == "review-stats":
        from . import teamreview
        teamreview.stats(S)
    elif a.cmd == "split":
        from . import split
        split.run(S, include_unreviewed=a.include_unreviewed)
    elif a.cmd == "export":
        from . import export
        export.run(S, include_unreviewed=a.include_unreviewed, no_images=a.no_images)
    elif a.cmd == "reverify":
        from . import reverify
        reverify.run(S, dry_run=a.dry_run)
    elif a.cmd == "qc":
        from . import qc
        return 0 if not qc.run(S)["hard_failures"] else 1
    elif a.cmd == "stats":
        from . import stats
        stats.run(S)
    elif a.cmd == "status":
        from .manifest import Manifest
        with Manifest(S.db_path) as db:
            table: dict[str, dict[str, int]] = {}
            for row in db.counts():
                table.setdefault(row["grp"], {})[row["status"]] = row["n"]
        cols = ["candidate", "screened", "segmented", "composed", "kept", "rejected"]
        print(f"{'group':<7}" + "".join(f"{c:>11}" for c in cols))
        for g, c in table.items():
            print(f"{g:<7}" + "".join(f"{c.get(k, 0):>11}" for k in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())

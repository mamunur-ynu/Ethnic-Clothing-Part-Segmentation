"""Load the collector's log.jsonl files (data/raw/<CODE>/log.jsonl) into the manifest."""
from __future__ import annotations

import json

from .manifest import Manifest
from .settings import Settings


def run(S: Settings, progress=print) -> dict:
    added = missing = 0
    with Manifest(S.db_path) as db, db.tx():
        for g in S.groups(include_disabled=True):
            log = S.raw_dir / g["code"] / "log.jsonl"
            if not log.exists():
                continue
            for line in log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if not r.get("image_id"):
                    continue                       # candidate that was not saved
                if not (S.raw_dir / r["file"]).exists():
                    missing += 1
                    continue
                rec = dict(r, grp=r["group"], search_query=r["query"], page_url=r["page_url"])
                added += db.insert(rec)
        # the same picture saved under two different groups -> label conflict, drop both
        dup = db.con.execute(
            "SELECT sha256 FROM images WHERE sha256 IS NOT NULL GROUP BY sha256 HAVING COUNT(DISTINCT grp) > 1"
        ).fetchall()
        for (sha,) in dup:
            db.con.execute("UPDATE images SET status='rejected', reason='conflicting_labels' WHERE sha256=?", (sha,))
        db.log("ingest", f"added={added} missing_files={missing} conflicting_sha={len(dup)}")
    progress(f"ingest: {added} new images, {missing} listed but missing on disk, "
             f"{len(dup)} pictures found under two groups (rejected)")
    return {"added": added, "missing": missing, "conflicts": len(dup)}

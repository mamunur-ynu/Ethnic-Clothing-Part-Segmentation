"""SQLite manifest: one row per image, updated by each stage.

Every stage reads and writes this single file (data/manifest.db), so the state of
any image can be answered with one query. ``status`` holds the latest pipeline
decision and ``reason`` explains a rejection.

status values
    candidate   ingested, waiting for screening
    screened    passed screening, waiting for SAM 3
    segmented   SAM 3 output present, waiting for compose
    composed    512x512 image + mask built, waiting for human review
    kept        human reviewer kept it
    rejected    removed (see reason)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id      TEXT PRIMARY KEY,
    grp           TEXT NOT NULL,
    file          TEXT NOT NULL,          -- path relative to data/raw
    search_query  TEXT,
    image_url     TEXT,
    page_url      TEXT,
    publisher     TEXT,
    page_title    TEXT,
    collected_at  TEXT,
    license       TEXT,
    verify_status TEXT,
    evidence      TEXT,
    sha256        TEXT,
    width         INTEGER,
    height        INTEGER,
    status        TEXT NOT NULL DEFAULT 'candidate',
    reason        TEXT,
    blur          REAL,
    dhash         TEXT,                   -- 16 hex chars, raw image
    final_dhash   TEXT,                   -- 64 hex chars, subject crop of the composed image
    seg_json      TEXT,                   -- summary returned by the SAM 3 stage
    center_dx     REAL,
    center_dy     REAL,
    fill          REAL,
    flags         TEXT,                   -- comma-separated warnings for the reviewer
    garment_type  TEXT,
    split         TEXT,
    cluster       TEXT,                   -- images sharing a cluster must share a split
    review        TEXT,                   -- keep / reject / NULL
    review_reason TEXT,
    reviewer      TEXT,
    reviewed_at   TEXT,
    evidence_type   TEXT,                 -- image_caption / page_text / search_title
    source_category TEXT                  -- shop / social / news_publisher / museum / ...
);
CREATE INDEX IF NOT EXISTS idx_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_grp ON images(grp);
-- One row per decision by one person. images.review keeps the final decision;
-- this table keeps every decision, so two reviewers can be compared (Cohen's kappa).
CREATE TABLE IF NOT EXISTS reviews (
    image_id  TEXT NOT NULL,
    reviewer  TEXT NOT NULL,
    decision  TEXT NOT NULL,              -- keep / reject
    reason    TEXT,
    fix       TEXT,                       -- mask problems the reviewer saw, e.g. "headwear_missing"
    ts        TEXT,
    PRIMARY KEY (image_id, reviewer)
);
CREATE TABLE IF NOT EXISTS events (
    ts       TEXT DEFAULT (datetime('now')),
    stage    TEXT,
    message  TEXT
);
"""

COLUMNS = (
    "image_id grp file search_query image_url page_url publisher page_title collected_at "
    "license verify_status evidence sha256 width height status reason blur dhash final_dhash "
    "seg_json center_dx center_dy fill flags garment_type split cluster review review_reason "
    "reviewer reviewed_at evidence_type source_category"
).split()

# Columns added after the first version. Old manifest files get them with ALTER TABLE.
NEW_COLUMNS = {"evidence_type": "TEXT", "source_category": "TEXT"}


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        have = {r[1] for r in self.con.execute("PRAGMA table_info(images)")}
        if have:                                   # an existing file: add any missing columns first
            for col, typ in NEW_COLUMNS.items():
                if col not in have:
                    self.con.execute(f"ALTER TABLE images ADD COLUMN {col} {typ}")
            self.con.commit()
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def tx(self):
        try:
            yield self.con
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    # ---- reads -------------------------------------------------------------
    def rows(self, where: str = "1=1", params: tuple = ()) -> list[sqlite3.Row]:
        return self.con.execute(f"SELECT * FROM images WHERE {where} ORDER BY image_id", params).fetchall()

    def get(self, image_id: str) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM images WHERE image_id=?", (image_id,)).fetchone()

    def counts(self) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT grp, status, COUNT(*) n FROM images GROUP BY grp, status ORDER BY grp, status"
        ).fetchall()

    def reasons(self) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT grp, reason, COUNT(*) n FROM images WHERE status='rejected' "
            "GROUP BY grp, reason ORDER BY grp, n DESC"
        ).fetchall()

    # ---- writes ------------------------------------------------------------
    def insert(self, rec: dict) -> bool:
        """Insert a new image; returns False if the id already exists."""
        cols = [c for c in rec if c in COLUMNS]
        sql = f"INSERT OR IGNORE INTO images ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        cur = self.con.execute(sql, [rec[c] for c in cols])
        return cur.rowcount == 1

    def update(self, image_id: str, **fields) -> None:
        bad = set(fields) - set(COLUMNS)
        if bad:
            raise KeyError(f"unknown columns {bad}")
        sets = ",".join(f"{k}=?" for k in fields)
        self.con.execute(f"UPDATE images SET {sets} WHERE image_id=?", [*fields.values(), image_id])

    def reject(self, image_id: str, reason: str) -> None:
        self.update(image_id, status="rejected", reason=reason)

    def log(self, stage: str, message: str) -> None:
        self.con.execute("INSERT INTO events(stage, message) VALUES (?, ?)", (stage, message))

    # ---- per-person review decisions ------------------------------------------
    def add_review(self, image_id: str, reviewer: str, decision: str, reason: str | None,
                   fix: str | None, ts: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO reviews(image_id, reviewer, decision, reason, fix, ts) "
            "VALUES (?,?,?,?,?,?)", (image_id, reviewer, decision, reason, fix, ts))

    def reviews(self, where: str = "1=1", params: tuple = ()) -> list[sqlite3.Row]:
        return self.con.execute(f"SELECT * FROM reviews WHERE {where} ORDER BY image_id, reviewer",
                                params).fetchall()

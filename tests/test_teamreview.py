"""Two-person review: plan, blind sessions, majority vote, kappa, fix list."""
from ethnoseg import compose, ingest, screen, teamreview
from ethnoseg.manifest import Manifest
from ethnoseg.parts import process_file
from tests.synthetic import ColorPredictor
from tests.test_pipeline import FakeCV2, S  # noqa: F401


def _composed(S):
    q = lambda *a: None
    ingest.run(S, progress=q)
    screen.run(S, progress=q)
    S.seg_dir.mkdir(parents=True, exist_ok=True)
    pred = ColorPredictor()
    with Manifest(S.db_path) as db, db.tx():
        for row in db.rows("status='screened'"):
            process_file(pred, S.raw_dir / row["file"], S.seg_dir, row["image_id"], S.classes,
                         S.raw["paint_order"], S.section("segment"))
            db.update(row["image_id"], status="segmented")
    compose.run(S, progress=q)


def test_kappa():
    assert teamreview.cohen_kappa(["keep"] * 4, ["keep"] * 4) == 1.0
    k = teamreview.cohen_kappa(["keep", "keep", "reject", "reject"], ["keep", "reject", "reject", "reject"])
    assert abs(k - 0.5) < 1e-9


def test_plan_sessions_and_stats(S):
    _composed(S)
    with Manifest(S.db_path) as db, db.tx():
        ids = [r["image_id"] for r in db.rows("status='composed'")]
        # an earlier single reviewer already decided the first 4
        for i in ids[:4]:
            db.update(i, review="keep", reviewer="reviewer_1", reviewed_at="t", status="kept")
    q = lambda *a: None
    p = teamreview.plan(S, ["A", "B"], overlap=0.5, audit=0.5, progress=q)
    assert p["copied"] == 4 and p["audit"] == 2
    with Manifest(S.db_path) as db:
        mine = {r[0] for r in db.con.execute("SELECT image_id FROM assignments WHERE reviewer='A'")}
        theirs = {r[0] for r in db.con.execute("SELECT image_id FROM assignments WHERE reviewer='B'")}
    both = mine & theirs
    assert len(both) == p["overlap"] + p["audit"] and mine | theirs <= set(ids)
    # A keeps everything and marks one mask problem; B rejects everything
    teamreview.session(S, "A", cv2=FakeCV2(["f", "1"] + ["k"] * len(mine)), progress=q)
    teamreview.session(S, "B", cv2=FakeCV2(["r", "2"] * len(theirs)), progress=q)
    with Manifest(S.db_path) as db:
        for i in both:
            r = db.get(i)
            votes = len(db.reviews("image_id=?", (i,)))
            if votes == 2:               # A keep vs B reject -> tie -> conflict
                assert r["status"] == "rejected" and r["review_reason"] == "review_conflict"
            else:                        # audit: reviewer_1 keep + A keep vs B reject -> keep
                assert votes == 3 and r["status"] == "kept"
        for i in theirs - mine:
            assert db.get(i)["status"] == "rejected" and db.get(i)["review_reason"] == "bad_cutout"
    st = teamreview.stats(S, progress=q)
    ab = [x for x in st["pairs"] if x["pair"] == "A vs B"][0]
    assert ab["images"] == len(both) and ab["agreement"] == 0.0
    assert st["fix_list_rows"] == 1
    assert (S.reports_dir / "fix_list.csv").read_text().count("headwear_missing") == 1

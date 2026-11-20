"""Review window: a held-down key must not skip through images."""
from ethnoseg import compose, ingest, review, screen
from ethnoseg.manifest import Manifest
from ethnoseg.parts import process_file
from tests.synthetic import ColorPredictor
from tests.test_pipeline import FakeCV2, S  # noqa: F401  (module-scoped fixture, own copy here)


class TimedCV2(FakeCV2):
    """Key presses with timestamps: (seconds since previous key, key)."""
    def __init__(self, timed):
        super().__init__([k for _, k in timed])
        self.gaps = [g for g, _ in timed]
        self.now = 0.0

    def waitKey(self, _):
        if self.gaps:
            self.now += self.gaps.pop(0)
        return super().waitKey(_)


def test_held_key_is_ignored(S):
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
    with Manifest(S.db_path) as db:
        a, b, c = [r["image_id"] for r in db.rows("status='composed' AND review IS NULL")][:3]
    fake = TimedCV2([(0.05, "k"),                     # too soon after the image appeared
                     (1.0, "k"),                      # fresh press after 1 s -> a kept
                     (0.03, "k"), (0.03, "k"),        # auto-repeat of a held key -> ignored
                     (0.03, "k"), (0.5, "k"),         # 0.5 s later: a fresh press -> b kept
                     (1.0, "q")])
    review.run(S, "person", cv2=fake, progress=q, min_view=0.4, clock=lambda: fake.now)
    with Manifest(S.db_path) as db:
        assert [x for x in (a, b, c) if db.get(x)["status"] == "kept"] == [a, b]

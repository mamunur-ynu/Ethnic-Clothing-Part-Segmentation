"""The demo's data functions work on an exported dataset (gradio itself is not needed here)."""
import app_demo
from ethnoseg import export, split
from ethnoseg.manifest import Manifest
from tests.test_pipeline import S  # noqa: F401
from tests.test_teamreview import _composed


def test_demo_functions(S):
    _composed(S)
    with Manifest(S.db_path) as db, db.tx():
        for r in db.rows("status='composed'"):
            db.update(r["image_id"], status="kept", review="keep", reviewer="x")
    q = lambda *a: None
    split.run(S, progress=q)
    export.run(S, progress=q)
    rows, classes = app_demo.load_index(S.dataset_dir)
    grp = rows[0]["ethnicity"]
    ids = app_demo.image_ids(rows, grp, rows[0]["split"])
    img, legend, src = app_demo.show(rows, classes, ids[0], S.dataset_dir)
    assert img.size[0] == 2 * img.size[1] and "Source page" in src and "|" in legend

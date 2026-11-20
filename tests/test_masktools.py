"""Mask rule check, editor painting, saving with the SAM 3 original kept, and agreement."""
import numpy as np
from PIL import Image

from ethnoseg import masktools
from ethnoseg.manifest import Manifest
from tests.test_pipeline import S  # noqa: F401
from tests.test_teamreview import _composed


def test_check_one():
    sub = np.zeros((50, 50), bool); sub[10:40, 10:40] = True
    lab = np.zeros((50, 50), np.uint8); lab[10:25, 10:40] = 3; lab[25:40, 10:40] = 2
    r = masktools.check_one(lab, sub, [1, 2, 3, 4, 5, 6])
    assert "dress_mixed_with_top_bottom" in r["flags"] and "label_outside_subject" not in r["flags"]
    lab[0, 0] = 1
    assert "label_outside_subject" in masktools.check_one(lab, sub, [1, 2, 3, 4, 5, 6])["flags"]
    lab[1, 1] = 9
    assert "bad_class_id" in masktools.check_one(lab, sub, [1, 2, 3, 4, 5, 6])["flags"]


def test_editor_paints_only_on_subject():
    sub = np.zeros((40, 40), bool); sub[:, :20] = True
    ed = masktools.Editor(np.zeros((40, 40), np.uint8), sub)
    ed.cls, ed.brush = 4, 5
    ed.paint(20, 20, start=True)
    assert (ed.lab[:, 20:] == 0).all() and (ed.lab[:, :20] == 4).any()
    ed.undo()
    assert (ed.lab == 0).all()


def test_save_agreement_sheets(S):
    _composed(S)
    with Manifest(S.db_path) as db, db.tx():
        ids = [r["image_id"] for r in db.rows("status='composed'")]
        for i in ids:
            db.update(i, status="kept", review="keep", reviewer="x")
    iid = ids[0]
    lab = np.asarray(Image.open(S.work_dir / "mask" / f"{iid}.png")).copy()
    orig = lab.copy()
    lab[lab == lab.max()] = 0                       # "correct" the mask: remove one class
    masktools.save_mask(S, iid, lab)
    assert (np.asarray(Image.open(S.work_dir / "mask_auto" / f"{iid}.png")) == orig).all()
    with Manifest(S.db_path) as db:
        assert "mask_edited" in db.get(iid)["flags"]
    r = masktools.agreement(S, progress=lambda *a: None)
    assert r["images_corrected"] == 1 and r["pixel_agreement"] < 1
    assert masktools.sheets(S, progress=lambda *a: None) > 0
    assert masktools.check(S, progress=lambda *a: None)["hard"] == 0

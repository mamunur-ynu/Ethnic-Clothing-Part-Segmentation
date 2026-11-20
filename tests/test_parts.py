import numpy as np

from ethnoseg import settings
from ethnoseg.parts import segment_with, build_label_map
from tests.synthetic import ColorPredictor, person_image

S = settings.load()
SEG = S.section("segment")
IDS = S.class_ids


def run(img, **kw):
    return segment_with(ColorPredictor(**kw), img, S.classes, S.raw["paint_order"], SEG)


def test_two_piece_outfit_dress_reassigned():
    label, info = run(person_image(1, outfit="two_piece"))
    assert info["status"] == "ok" and info["two_piece"]
    present = set(np.unique(label).tolist())
    assert IDS["dress"] not in present                       # confused 'dress' pixels went to top/bottom
    for name in ("top", "bottom", "headwear", "footwear", "waistwear"):
        assert IDS[name] in present, name
    assert 255 in present                                    # skin stays 'subject, not clothing'
    # apron painted over the top/bottom it overlaps
    assert info["part_share"]["waistwear"] > 0.02


def test_robe_kept_as_dress():
    label, info = run(person_image(2, outfit="robe"))
    assert info["status"] == "ok" and not info["two_piece"]
    assert info["part_share"]["dress"] > 0.3
    assert info["part_share"]["top"] == 0 and info["part_share"]["bottom"] == 0


def test_hat_outside_person_mask_joins_subject():
    label, _ = run(person_image(3))
    ys = np.nonzero(label == IDS["headwear"])[0]
    assert len(ys) > 0 and ys.min() < np.nonzero(label == 255)[0].min()


def test_multiple_subjects_rejected():
    label, info = run(person_image(4, w=760, second_person=True))
    assert label is None and info["status"] == "multiple_subjects"


def test_no_subject():
    img = person_image(5)
    label, info = segment_with(lambda im, prompts: {p: [] for p in prompts}, img, S.classes,
                               S.raw["paint_order"], SEG)
    assert label is None and info["status"] == "no_subject"


def test_part_far_from_subject_ignored():
    subj = np.zeros((100, 100), bool)
    subj[10:60, 10:40] = True
    far = np.zeros_like(subj)
    far[80:95, 80:95] = True
    label, s = build_label_map(subj, {"top": [(far, 0.9)]}, S.classes, S.raw["paint_order"], SEG)
    assert (label == IDS["top"]).sum() == 0


def test_neighbour_parts_not_attached():
    """A second mannequin's top inside the expanded box but not touching the person is ignored."""
    subj = np.zeros((200, 200), bool)
    subj[20:180, 60:120] = True                       # main person
    own_top = np.zeros_like(subj)
    own_top[40:100, 60:120] = True
    neighbour = np.zeros_like(subj)
    neighbour[40:100, 126:131] = True                 # 6 px away, inside the 10% box growth
    label, s = build_label_map(subj, {"top": [(own_top, 0.9), (neighbour, 0.9)]}, S.classes,
                               S.raw["paint_order"], SEG)
    assert (label[:, 124:] > 0).sum() == 0
    assert s["rules"] >= 2


def test_dilate():
    from ethnoseg.parts import dilate
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert dilate(m, 2).sum() == 25 and dilate(m, 0).sum() == 1

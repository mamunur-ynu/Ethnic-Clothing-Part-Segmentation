"""End-to-end offline run: synthetic collection -> ingest -> screen -> segment (colour fake
of SAM 3, same per-image code as the Modal job) -> compose -> review (scripted keys)
-> split -> export -> qc -> stats, then planted defects that QC must catch."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageFilter

from ethnoseg import compose, export, ingest, qc, review, screen, settings, split, stats
from ethnoseg.manifest import Manifest
from ethnoseg.parts import process_file
from tests.synthetic import ColorPredictor, jpeg, person_image

ROOT = Path(__file__).resolve().parent.parent
N_PER_GROUP = 24


class FakeCV2:
    """Stands in for OpenCV's window functions; replays a list of key presses."""
    WINDOW_NORMAL = 0

    def __init__(self, keys):
        self.keys = list(keys)
        self.shown = 0

    def namedWindow(self, *a):
        pass

    def imshow(self, name, img):
        assert img.ndim == 3 and img.shape[2] == 3
        self.shown += 1

    def waitKey(self, _):
        return ord(self.keys.pop(0)) if self.keys else ord("q")

    def destroyAllWindows(self):
        pass


def write_raw(S, code, n, rows, img, page):
    d = S.raw_dir / code / "img"
    d.mkdir(parents=True, exist_ok=True)
    iid = f"{code}{n:05d}"
    data = jpeg(img)
    (d / f"{iid}.jpg").write_bytes(data)
    rows.append({"image_id": iid, "group": code, "file": f"{code}/img/{iid}.jpg", "query": "q",
                 "image_url": f"http://img.example/{iid}.jpg", "page_url": page, "title": "",
                 "publisher": "example.org", "page_title": "t", "verify_status": "verified_page",
                 "evidence": f"... {S.group(code)['zh']} 服饰 ...", "collected_at": "2026-11-19T00:00:00Z",
                 "license": "unknown", "sha256": hashlib.sha256(data).hexdigest(), "width": img.width,
                 "height": img.height})
    return data


@pytest.fixture(scope="module")
def S(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["qc"].update(min_per_group=15, min_total=80)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    S = settings.load(root / "config.yaml")
    seed = 0
    for gi, g in enumerate(S.groups()):
        rows = []
        for n in range(1, N_PER_GROUP + 1):
            seed += 1
            outfit = "robe" if (gi + n) % 3 == 0 else "two_piece"
            write_raw(S, g["code"], n, rows, person_image(seed, outfit=outfit),
                      f"http://site.example/{g['code']}/page{(n + 1) // 2}")      # 2 images per page
        k = N_PER_GROUP
        base = person_image(1000 + gi)
        write_raw(S, g["code"], k + 1, rows, base.filter(ImageFilter.GaussianBlur(12)), "http://b/1")  # blurry
        write_raw(S, g["code"], k + 2, rows, person_image(2000 + gi, w=250, h=280), "http://b/2")     # small
        src = person_image(3000 + gi)
        write_raw(S, g["code"], k + 3, rows, src, "http://b/3")
        write_raw(S, g["code"], k + 4, rows, src.resize((540, 810)), "http://b/4")                    # near-dup
        write_raw(S, g["code"], k + 5, rows, person_image(4000 + gi, w=760, second_person=True), "http://b/5")
        if g["code"] in ("YI", "BA"):                                                                   # same photo, 2 groups
            write_raw(S, g["code"], k + 6, rows, person_image(5000), "http://b/6")
        with open(S.raw_dir / g["code"] / "log.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"image_url": "x", "group": g["code"], "reason": "unverified"}) + "\n")
    return S


def status_of(S, iid):
    with Manifest(S.db_path) as db:
        r = db.get(iid)
        return r["status"], r["reason"]


def test_full_pipeline(S):
    q = lambda *a: None
    k = N_PER_GROUP
    r = ingest.run(S, progress=q)
    assert r["added"] == 5 * (k + 5) + 2 and r["conflicts"] == 1
    assert status_of(S, f"YI{k + 6:05d}") == ("rejected", "conflicting_labels")

    screen.run(S, progress=q)
    assert status_of(S, f"YI{k + 1:05d}") == ("rejected", "blurry")
    assert status_of(S, f"YI{k + 2:05d}") == ("rejected", "too_small")
    st3, st4 = status_of(S, f"YI{k + 3:05d}"), status_of(S, f"YI{k + 4:05d}")
    assert st3[0] == "screened" and st4 == ("rejected", f"near_duplicate_of:YI{k + 3:05d}")
    # re-running screen changes nothing
    screen.run(S, progress=q)
    assert status_of(S, f"YI{k + 3:05d}")[0] == "screened"

    # "GPU" stage: same per-image function the Modal job runs, colour-based fake SAM 3
    S.seg_dir.mkdir(parents=True, exist_ok=True)
    pred = ColorPredictor()
    with Manifest(S.db_path) as db, db.tx():
        for row in db.rows("status='screened'"):
            out = process_file(pred, S.raw_dir / row["file"], S.seg_dir, row["image_id"], S.classes,
                               S.raw["paint_order"], S.section("segment"))
            assert out["status"] in ("ok", "multiple_subjects"), out
            db.update(row["image_id"], status="segmented")

    c = compose.run(S, progress=q)
    assert status_of(S, f"YI{k + 5:05d}") == ("rejected", "multiple_subjects")
    assert c.get("composed", 0) >= 5 * (k - 2), c
    img = np.asarray(Image.open(S.work_dir / "img" / "YI00001.png"))
    sub = np.asarray(Image.open(S.work_dir / "subject" / "YI00001.png")) > 127
    assert img.shape == (512, 512, 3) and (img[~sub] == 255).all()

    # review: reject YI00001 (wrong parts), test 'back' + change, keep the rest
    keys = ["r", "1", "k", "b", "r", "x", "s"]           # r+1 reject; keep; back; r then cancel; skip
    fake = FakeCV2(keys)
    review.run(S, "tester", group="YI", cv2=fake, progress=q)
    assert status_of(S, "YI00001") == ("rejected", "review:wrong_parts")
    assert status_of(S, "YI00002")[0] == "kept"
    fake = FakeCV2(["k"] * 500)
    review.run(S, "tester", cv2=fake, progress=q)
    with Manifest(S.db_path) as db:
        assert not db.rows("status='composed'")

    split.run(S, progress=q)
    export.run(S, progress=q)
    res = qc.run(S, progress=q)
    assert not res["hard_failures"], res
    d = S.dataset_dir
    meta = list(csv.DictReader(open(d / "metadata.csv", encoding="utf-8-sig")))
    assert {m["review"] for m in meta} == {"keep"}
    pages = {}
    for m in meta:
        pages.setdefault(m["source_url"], set()).add(m["split"])
    assert all(len(v) == 1 for v in pages.values())
    assert json.loads((d / "annotations" / "classes.json").read_text())["6"] == "waistwear"
    lab = list(csv.DictReader(open(d / "annotations" / "classification_labels.csv", encoding="utf-8-sig")))
    assert lab[0].keys() == {"image_id", "path", "ethnicity", "garment_type", "split"}
    assert (d / "README.md").read_text(encoding="utf-8").count("|") > 20

    stats.run(S, progress=q)
    for f in ("images_per_group.png", "part_share_per_group.png", "rejection_reasons.png", "funnel.png",
              "source_categories.png"):
        assert (S.reports_dir / "figures" / f).exists(), f
    assert (S.reports_dir / "private" / "examples.jpg").exists()      # photos: kept out of figures/
    summ = json.loads((S.reports_dir / "summary.json").read_text(encoding="utf-8"))
    assert summ["funnel"]["final dataset"] == len(meta) and "class_pixel_share_of_clothing" in summ

    # ---- planted defects: each must be caught by QC ---------------------------------
    m0, m1 = meta[0], meta[1]
    im = np.asarray(Image.open(d / m0["path"])).copy()
    im[0, 0] = (250, 250, 250)
    Image.fromarray(im).save(d / m0["path"])
    mk = np.asarray(Image.open(d / m1["mask_path"])).copy()
    mk[256, 256] = 9
    Image.fromarray(mk).save(d / m1["mask_path"])
    other = next(m for m in meta[3:] if m["split"] != meta[2]["split"])
    shutil.copy(d / meta[2]["path"], d / other["path"])                # same picture in two splits
    shutil.copy(S.work_dir / "subject" / f"{meta[2]['image_id']}.png",
                S.work_dir / "subject" / f"{other['image_id']}.png")
    res = qc.run(S, progress=q)
    hf = res["hard_failures"]
    assert hf.get("background_not_pure_white") == 1
    assert hf.get("mask_bad_class_id") == 1
    assert hf.get("near_duplicate_across_splits", 0) >= 1


def test_no_images_export(S):
    export.run(S, no_images=True, progress=lambda *a: None)
    assert not list((S.dataset_dir / "images").rglob("*.png"))
    assert list((S.dataset_dir / "annotations" / "masks").rglob("*.png"))
    res = qc.run(S, progress=lambda *a: None)
    assert res["images_checked_on_disk"] == 0 and not res["hard_failures"]

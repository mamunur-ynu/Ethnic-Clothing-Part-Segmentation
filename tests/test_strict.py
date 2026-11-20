"""Strict one-group evidence rule, source categories and the manifest upgrade."""
import sqlite3

from ethnoseg.manifest import Manifest
from ethnoseg.web import groups_named, source_category, strict_evidence

ZH = {"code": "ZH", "zh": "壮族", "zh_aliases": [], "en_patterns": [r"\bzhuang (ethnic|minority)"]}
UY = {"code": "UY", "zh": "维吾尔族", "zh_aliases": ["维吾尔", "维族"], "en_patterns": [r"\buy(g|gh)ur\b"]}
MA = {"code": "MA", "zh": "满族", "zh_aliases": [], "en_patterns": [r"\bmanchu\b"]}


def test_strict_rule():
    assert strict_evidence(ZH, "壮族服饰 女 三月三", "") == (True, None)
    assert strict_evidence(ZH, "壮族 苗族 瑶族 土家族 演出服", "") == (False, "multi_group_evidence")
    assert strict_evidence(ZH, "少数民族演出服", "舞蹈服装") == (False, "own_group_missing")
    assert strict_evidence(UY, "新疆维吾尔舞蹈服装", "") == (True, None)       # alias without 族
    assert strict_evidence(UY, "维族 and Tibetan dress", "") == (False, "multi_group_evidence")
    # 内蒙古 is a place name, not the Mongolian group
    assert strict_evidence(MA, "内蒙古 满族旗袍", "") == (True, None)
    assert groups_named("Miao and Manchu costumes") == {"苗族", "满族"}


def test_source_category():
    assert source_category("https://item.taobao.com/item.htm?id=1") == "shop"
    assert source_category("https://mobile.yangkeduo.com/goods.html") == "shop"
    assert source_category("https://baijiahao.baidu.com/s?id=1") == "social"
    assert source_category("https://baike.baidu.com/item/x") == "reference"
    assert source_category("https://www.sohu.com/a/1") == "news_publisher"
    assert source_category("https://www.ynmuseum.org/x") == "museum"
    assert source_category("https://www.gxzf.gov.cn/x") == "institution"
    assert source_category("https://example.org/x") == "other"


def test_manifest_upgrade(tmp_path):
    # an old manifest without the new columns gets them added on open
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id TEXT PRIMARY KEY, grp TEXT NOT NULL, file TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'candidate')")
    con.execute("INSERT INTO images VALUES ('ZH00001','ZH','x.jpg','kept')")
    con.commit(); con.close()
    with Manifest(db) as m:
        cols = {r[1] for r in m.con.execute("PRAGMA table_info(images)")}
        assert {"evidence_type", "source_category"} <= cols
        m.add_review("ZH00001", "A", "keep", None, None, "t")
        assert len(m.reviews()) == 1

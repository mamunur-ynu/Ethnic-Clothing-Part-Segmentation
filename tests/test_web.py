"""Collector test against a local fake 'Bing' + fake source pages (no internet)."""
import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from ethnoseg import web
from tests.synthetic import jpeg, person_image

GROUP = {"code": "YI", "name": "Yi", "zh": "彝族", "zh_aliases": [],
         "en_patterns": [r"\byi (ethnic|minority|people|costume|clothing|dress)"],
         "queries": ["彝族服饰", "Yi costume"]}
CFG = {"engines": ["so360", "baidu", "bing"], "max_results_per_query": 50, "min_side_px": 300, "max_image_mb": 5, "page_timeout_s": 5,
       "image_timeout_s": 5, "workers": 4, "max_groups_named_on_page": 2}


def make_site(port):
    big = jpeg(person_image(1))
    big2 = jpeg(person_image(2))
    small = jpeg(person_image(3, w=200, h=250))
    base = f"http://127.0.0.1:{port}"
    pages = {
        "/p/ok": "<html><head><title>凉山彝族服饰展</title></head><body>彝族传统服饰 女装</body></html>",
        "/p/caption": f"<html><body><p>民族服装合集</p><img src='{base}/img/cap.jpg' alt='彝族盛装 少女'></body></html>",
        "/p/list": "<html><body>56个民族服饰：彝族 苗族 白族 傣族 壮族 藏族</body></html>",
        "/p/none": "<html><body>漂亮的裙子 夏季新款</body></html>",
        "/p/en": "<html><body>A Yi ethnic minority woman in Liangshan</body></html>",
        "/p/gbk": "彝族服饰".encode("gbk"),
    }
    images = {"/img/a.jpg": big, "/img/cap.jpg": big2, "/img/small.jpg": small, "/img/bad.jpg": b"not an image",
              "/img/dup.jpg": big, "/img/list.jpg": big2, "/img/none.jpg": big, "/img/en.jpg": jpeg(person_image(4)),
              "/img/title.jpg": jpeg(person_image(5)), "/img/gbk.jpg": jpeg(person_image(6))}
    results = [("/img/a.jpg", "/p/ok", "彝族"), ("/img/cap.jpg", "/p/caption", ""),
               ("/img/small.jpg", "/p/ok", ""), ("/img/bad.jpg", "/p/ok", ""), ("/img/dup.jpg", "/p/ok", ""),
               ("/img/list.jpg", "/p/list", "彝族服饰"), ("/img/none.jpg", "/p/none", "女装"),
               ("/img/en.jpg", "/p/en", ""), ("/img/title.jpg", "/p/missing", "彝族姑娘服装"),
               ("/img/gbk.jpg", "/p/gbk", "")]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/images/async":                     # Bing: 4 results
                first = int(parse_qs(u.query)["first"][0])
                rows = results[:4] if first == 1 else []
                html = "".join(
                    "<a class='iusc' m='{}'></a>".format(
                        json.dumps({"murl": base + i, "purl": base + p, "t": t}).replace("'", "&#39;"))
                    for i, p, t in rows)
                body, ctype = html.encode(), "text/html; charset=utf-8"
            elif u.path == "/j":                              # 360: results 4..7, two pages
                sn = int(parse_qs(u.query)["sn"][0])
                rows = results[4:6] if sn == 0 else results[6:8] if sn == 2 else []
                body = json.dumps({"list": [{"img": base + i, "link": base + p, "title": f"<em>{t}</em>",
                                             "qhimg_url": base + "/cache" + i}
                                            for i, p, t in rows]}).encode()
                ctype = "application/json"
            elif u.path == "/search/acjson":                  # Baidu: results 8.., image behind a proxy URL
                pn = int(parse_qs(u.query)["pn"][0])
                rows = results[8:] if pn == 0 else []
                from urllib.parse import quote
                body = json.dumps({"data": [{"replaceUrl": [{"ObjURL": f"{base}/image_search/src={quote(base + i, safe='')}&refer=x",
                                                             "FromURL": base + p}], "fromPageTitleEnc": t}
                                            for i, p, t in rows] + [{}]}).encode()
                ctype = "application/json"
            elif u.path.startswith("/image_search/"):
                self.send_response(404)
                self.end_headers()
                return
            elif u.path in pages:
                v = pages[u.path]
                body = v if isinstance(v, bytes) else v.encode()
                ctype = "text/html" + ("" if isinstance(v, bytes) else "; charset=utf-8")
                if u.path == "/p/gbk":
                    body = b"<html><head><meta charset='gbk'></head><body>" + v + b"</body></html>"
            elif u.path.startswith("/cache/img/"):              # the engine's cached copy
                if u.path == "/cache/img/en.jpg" and self.server.fail_cache:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, ctype = images[u.path[len("/cache"):]], "image/jpeg"
            elif u.path == "/img/en.jpg":                         # original is hot-link protected
                body, ctype = b"<html>forbidden</html>", "text/html"
            elif u.path in images:
                body, ctype = images[u.path], "image/jpeg"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return H


SITE_SERVERS = {}


@pytest.fixture
def site():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    srv.RequestHandlerClass = make_site(srv.server_address[1])
    srv.fail_cache = False
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    SITE_SERVERS[url] = srv
    yield url
    srv.shutdown()


def test_collect_group(site, tmp_path):
    urls = {"bing": site + "/images/async?q={q}&first={offset}", "so360": site + "/j?q={q}&sn={offset}",
            "baidu": site + "/search/acjson?word={q}&pn={offset}"}
    s = web.collect_group(GROUP, CFG, tmp_path, urls=urls, pause=0, progress=lambda *_: None)
    recs = [json.loads(l) for l in (tmp_path / "YI" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {r["image_url"].rsplit("/", 1)[1]: r for r in recs}
    assert len(recs) == 10                               # second query returned the same URLs -> skipped
    assert by["a.jpg"]["verify_status"] == "verified_page"
    assert "彝族" in by["a.jpg"]["evidence"]
    assert by["cap.jpg"]["verify_status"] == "verified_caption"
    assert by["small.jpg"]["reason"] == "too_small"
    assert by["bad.jpg"]["reason"] == "corrupt_or_not_image"
    # a.jpg and dup.jpg are the same bytes: exactly one is saved, the other is an exact duplicate
    pair = [by["a.jpg"], by["dup.jpg"]]
    assert sorted(bool(r.get("image_id")) for r in pair) == [False, True]
    assert [r.get("reason") for r in pair if not r.get("image_id")] == ["exact_duplicate"]
    assert by["list.jpg"]["verify_status"] == "ambiguous" and "image_id" not in by["list.jpg"]
    assert by["none.jpg"]["verify_status"] == "unverified" and "image_id" not in by["none.jpg"]
    assert by["en.jpg"]["verify_status"] == "verified_page"
    assert by["en.jpg"]["downloaded_from"].endswith("/cache/img/en.jpg")   # original failed, cached copy used
    assert by["title.jpg"]["verify_status"] == "verified_title"
    assert by["gbk.jpg"]["verify_status"] == "verified_page"      # GBK-encoded page decoded correctly
    saved = sorted(p.name for p in (tmp_path / "YI" / "img").iterdir())
    assert saved == [f"YI{n:05d}.jpg" for n in range(1, 6)] and s["saved"] == 5
    # resumable: second run finds nothing new and keeps numbering
    s2 = web.collect_group(GROUP, CFG, tmp_path, urls=urls, pause=0, progress=lambda *_: None)
    assert s2["candidates"] == 0 and s2["saved"] == 0


def test_verify_rules():
    hit = web.Hit("http://x/i.jpg", "http://x/p", "", "q")
    assert web.verify(hit, GROUP, "<p>yi ethnic dress</p>".encode(), 2)["verify_status"] == "verified_page"
    assert web.verify(hit, GROUP, "<p>Yiwu market dress</p>".encode(), 2)["verify_status"] == "unverified"
    two = "<p>彝族 和 白族 服饰对比</p>".encode()
    assert web.verify(hit, GROUP, two, 2)["verify_status"] == "verified_page"
    assert web.verify(hit, GROUP, None, 2)["verify_status"] == "unverified"


def test_retry_failed(site, tmp_path):
    """A download that failed is retried with --retry-failed and then saved."""
    urls = {"bing": site + "/images/async?q={q}&first={offset}", "so360": site + "/j?q={q}&sn={offset}",
            "baidu": site + "/search/acjson?word={q}&pn={offset}"}
    q = lambda *a: None
    # first run with the cache broken -> en.jpg fails
    from ethnoseg import web as w
    srv = SITE_SERVERS[site]
    srv.fail_cache = True
    s1 = w.collect_group(GROUP, CFG, tmp_path, urls=urls, pause=0, progress=q)
    recs = [json.loads(l) for l in (tmp_path / "YI" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("reason") for r in recs if r["image_url"].endswith("/en.jpg")] == ["corrupt_or_not_image"]
    srv.fail_cache = False
    s2 = w.collect_group(GROUP, CFG, tmp_path, urls=urls, pause=0, progress=q)
    assert s2["candidates"] == 0                                   # normal re-run skips it
    s3 = w.collect_group(GROUP, CFG, tmp_path, urls=urls, pause=0, progress=q, retry_failed=True)
    assert s3["saved"] == 1                                        # retried and saved

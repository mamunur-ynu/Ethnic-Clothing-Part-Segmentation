"""Image search (360 / Baidu / Bing), download and source-page verification.

Runs inside a Modal CPU container (see modal_collect.py) but has no Modal
dependency, so it can also be run and tested locally.

For each candidate the source page is fetched and the group's name is looked up:
    verified_caption  the <img> alt/title for this exact image names the group
    verified_page     the page text names the group and at most N groups in total
    verified_title    only Bing's result title names it (page unreachable or silent)
    ambiguous         page names the group but also 3+ other groups (listicles), or the
                      page title + evidence name another group too (strict rule)
    unverified        group not named anywhere -> not downloaded
Only verified_* candidates are saved. The label is never inferred from the image.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

# The 56 recognised ethnic groups (official Chinese names), used to spot list pages.
ETHNIC_56 = (
    "汉族 蒙古族 回族 藏族 维吾尔族 苗族 彝族 壮族 布依族 朝鲜族 满族 侗族 瑶族 白族 土家族 哈尼族 "
    "哈萨克族 傣族 黎族 傈僳族 佤族 畲族 高山族 拉祜族 水族 东乡族 纳西族 景颇族 柯尔克孜族 土族 "
    "达斡尔族 仫佬族 羌族 布朗族 撒拉族 毛南族 仡佬族 锡伯族 阿昌族 普米族 塔吉克族 怒族 乌孜别克族 "
    "俄罗斯族 鄂温克族 德昂族 保安族 裕固族 京族 塔塔尔族 独龙族 鄂伦春族 赫哲族 门巴族 珞巴族 基诺族"
).split()

# Common short forms that also refer to one group. Only forms that are safe in normal text
# are listed (e.g. not 蒙古, which is also part of the province name 内蒙古).
ZH_ALIASES = {"维吾尔族": ["维吾尔", "维族"], "蒙古族": ["蒙族"]}

# English names that clearly mean one group (short ones like "Yi", "Bai", "Han", "Li", "Dai"
# are left out because they are also common words or surnames).
EN_GROUPS = {
    "苗族": r"\b(miao|hmong)\b", "藏族": r"\btibetan\b", "蒙古族": r"\bmongolian\b",
    "维吾尔族": r"\b(uyghur|uygur|uighur)\b", "壮族": r"\bzhuang\b", "满族": r"\bmanchu\b",
    "土家族": r"\btujia\b", "哈尼族": r"\bhani\b", "哈萨克族": r"\bkazak[h]?\b", "纳西族": r"\bnaxi\b",
    "布依族": r"\bbouyei\b", "傈僳族": r"\blisu\b", "景颇族": r"\bjingpo\b", "羌族": r"\bqiang\b",
}
_EN_RX = {g: re.compile(p, re.I) for g, p in EN_GROUPS.items()}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
LICENSE_UNKNOWN = "unknown - all rights reserved assumed; do not redistribute"


@dataclass
class Hit:
    image_url: str
    page_url: str
    title: str
    query: str
    fallbacks: tuple = ()      # the engine's own cached copies, tried if image_url fails


# --------------------------------------------------------------------------- search
# Three engines. From Modal's servers (Nov 2026) 360 and Baidu answer with JSON that
# includes each image's source page; Bing often returns pages without result tiles.
ENGINE_URLS = {
    "so360": "https://image.so.com/j?q={q}&sn={offset}&pn=60",
    "baidu": ("https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ie=utf-8&oe=utf-8"
              "&word={q}&queryWord={q}&pn={offset}&rn=30"),
    "bing": "https://www.bing.com/images/async?q={q}&first={offset}&count=35&mmasync=1&setlang=zh-cn",
}
_TAG = re.compile(r"<[^>]+>")


def _clean(t) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", str(t or ""))).strip()


def _unwrap_baidu(url: str) -> str:
    """gimg2.baidu.com/image_search/src=<original>&... -> original image URL."""
    if "image_search/src=" in url:
        from urllib.parse import parse_qs, unquote
        q = parse_qs(urlparse(url).path.split("image_search/", 1)[1] + "&" + urlparse(url).query)
        if q.get("src"):
            return unquote(q["src"][0])
    return url


def parse_bing(html: str, query: str) -> list[Hit]:
    soup = BeautifulSoup(html, "html.parser")
    hits = []
    for a in soup.select("a.iusc"):
        try:
            m = json.loads(a.get("m", ""))
        except ValueError:
            continue
        murl, purl = m.get("murl"), m.get("purl")
        if murl and purl and murl.startswith("http") and purl.startswith("http"):
            hits.append(Hit(murl, purl, _clean(m.get("t")), query))
    return hits


def parse_so360(text: str, query: str) -> list[Hit]:
    try:
        items = json.loads(text).get("list") or []
    except ValueError:
        return []
    hits = []
    for it in items:
        img, page = it.get("img") or it.get("qhimg_url"), it.get("link")
        if img and page and str(img).startswith("http") and str(page).startswith("http"):
            alt = tuple(u for u in (it.get("qhimg_url"), it.get("_thumb_bak"), it.get("_thumb"))
                        if isinstance(u, str) and u.startswith("http") and u != img)
            hits.append(Hit(img, page, _clean(it.get("title")), query, alt))
    return hits


def parse_baidu(text: str, query: str) -> list[Hit]:
    try:
        items = json.loads(text, strict=False).get("data") or []
    except ValueError:
        return []
    hits = []
    for it in items:
        if not it:
            continue
        rep = (it.get("replaceUrl") or [{}])[-1]
        proxy = rep.get("ObjURL") or ""
        img = _unwrap_baidu(proxy or it.get("middleURL") or "")
        page = rep.get("FromURL") or ""
        title = it.get("fromPageTitleEnc") or it.get("fromPageTitle") or ""
        if img.startswith("http") and page.startswith("http"):
            alt = tuple(u for u in (proxy, it.get("middleURL"), it.get("hoverURL"))
                        if isinstance(u, str) and u.startswith("http") and u != img)
            hits.append(Hit(img, page, _clean(title), query, alt))
    return hits


PARSERS = {"so360": parse_so360, "baidu": parse_baidu, "bing": parse_bing}
PAGE_START = {"so360": 0, "baidu": 0, "bing": 1}


def search(query: str, limit: int, session: requests.Session, engine: str = "so360",
           urls: dict | None = None, pause: float = 0.5) -> list[Hit]:
    """Page through one engine until `limit` hits or no new results."""
    url_tpl = (urls or ENGINE_URLS)[engine]
    seen, out, offset = set(), [], PAGE_START[engine]
    headers = dict(HEADERS)
    if engine == "baidu":
        headers["Referer"] = "https://image.baidu.com/"
        try:
            session.get(url_tpl.split("/search/")[0] + "/", headers=HEADERS, timeout=15)   # cookies
        except requests.RequestException:
            pass
    while len(out) < limit:
        try:
            r = session.get(url_tpl.format(q=quote_plus(query), offset=offset), headers=headers, timeout=20)
            r.raise_for_status()
        except requests.RequestException:
            break
        page = PARSERS[engine](r.text, query)
        new = [h for h in page if h.image_url not in seen]
        if not new:
            break
        for h in new:
            seen.add(h.image_url)
            out.append(h)
        offset += len(page)
        time.sleep(pause)
    return out[:limit]


# --------------------------------------------------------------------------- verification
def group_matcher(group: dict):
    zh_terms = [group["zh"], *group.get("zh_aliases", [])]
    en_res = [re.compile(p, re.I) for p in group.get("en_patterns", [])]

    def find(text: str) -> tuple[int, int] | None:
        """Return (start, end) of the first mention of the group, or None."""
        best = None
        for t in zh_terms:
            i = text.find(t)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, i + len(t))
        for rx in en_res:
            m = rx.search(text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end())
        return best

    return find


def snippet(text: str, span: tuple[int, int], width: int = 30) -> str:
    s = text[max(0, span[0] - width): span[1] + width]
    return re.sub(r"\s+", " ", s).strip()


def groups_named(text: str) -> set[str]:
    """All of the 56 groups named in the text (official name, safe alias or English name)."""
    found = {g for g in ETHNIC_56 if g in text}
    found |= {g for g, alts in ZH_ALIASES.items() if any(a in text for a in alts)}
    found |= {g for g, rx in _EN_RX.items() if rx.search(text)}
    return found


def strict_evidence(group: dict, page_title: str, evidence: str) -> tuple[bool, str | None]:
    """Strict label rule, checked on the page title + the evidence text together.

    The text must name this image's own group, and no other of the 56 groups.
    Returns (True, None) or (False, "own_group_missing" / "multi_group_evidence").
    """
    text = f"{page_title or ''} {evidence or ''}"
    if group_matcher(group)(text) is None:
        return False, "own_group_missing"
    others = groups_named(text) - {group["zh"]}
    if others:
        return False, "multi_group_evidence"
    return True, None


EVIDENCE_TYPE = {"verified_caption": "image_caption", "verified_page": "page_text",
                 "verified_title": "search_title"}

# Source category from the page's web address. Checked in this order; first match wins.
SOURCE_RULES = [
    ("shop", ("taobao.com", "tmall.com", "jd.com", "yangkeduo.com", "pinduoduo.com", "1688.com",
              "alibaba.com", "suning.com", "vip.com", "dangdang.com", "weidian.com", "17zwd.com",
              "made-in-china.com", "aliexpress.com", "dhgate.com", "amazon.", "b2b.baidu.com")),
    ("museum", ("museum", "bowuguan", "dpm.org.cn", "chnmuseum")),
    ("institution", (".gov.cn", ".edu.cn", "ihchina.cn", "chinaculture", "cxbw.org.cn")),
    ("reference", ("baike.baidu.com", "wikipedia.org", "baike.so.com", "freeman-pedia")),
    ("news_publisher", ("sohu.com", "sina.com.cn", "163.com", "qq.com", "people.com.cn", "xinhuanet",
                        "chinanews", "cnr.cn", "cctv", "thepaper.cn", "gxnews", "yunnan.cn", "yndaily",
                        "scol.com.cn", "kunming.cn", "china.com.cn", "cjn.cn", "thecover.cn",
                        "chinadaily", "ifeng.com", "china.org.cn", "360kuai.com")),
    ("stock_photo", ("huitu.com", "58pic.com", "alamy.com", "vcg.com", "shetu66.com", "tukuppt.com",
                     "aigei.com", "nipic.com", "699pic.com", "quanjing.com", "veer.com", "dreamstime",
                     "shutterstock", "gettyimages")),
    ("social", ("douyin.com", "weibo.", "xiaohongshu.com", "zhihu.com", "bilibili.com", "douban.com",
                "huaban.com", "duitang.com", "tuchong.com", "meipian.cn", "baijiahao.baidu.com",
                "mbd.baidu.com", "quanmin.baidu.com", "zhidao.baidu.com", "tieba.baidu.com",
                "dianping.com", "iqiyi.com", "youtube.com", "zcool.com.cn", "lofter.com",
                "xueqiu.com", "kuaishou.com", "toutiao.com")),
]


def source_category(page_url: str) -> str:
    host = urlparse(page_url or "").netloc.lower()
    for cat, keys in SOURCE_RULES:
        if any(k in host for k in keys):
            return cat
    return "other"


def _same_image(src: str, image_url: str) -> bool:
    if not src:
        return False
    a, b = urlparse(src).path.rstrip("/"), urlparse(image_url).path.rstrip("/")
    return bool(a) and bool(b) and (a == b or a.endswith(b.split("/")[-1]) and len(b.split("/")[-1]) > 8)


def verify(hit: Hit, group: dict, html: bytes | None, max_groups: int) -> dict:
    """Decide verify_status for one candidate from its page HTML (bytes or None)."""
    find = group_matcher(group)
    publisher = urlparse(hit.page_url).netloc
    page_title = ""
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()
        site = soup.find("meta", attrs={"property": "og:site_name"})
        if site and site.get("content"):
            publisher = site["content"].strip()
        # caption = alt/title of the <img> that is this exact image
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original") or ""
            if _same_image(src, hit.image_url):
                cap = " ".join(filter(None, [img.get("alt"), img.get("title")]))
                span = find(cap)
                if span:
                    return dict(verify_status="verified_caption", evidence=snippet(cap, span),
                                publisher=publisher, page_title=page_title)
        text = page_title + " " + soup.get_text(" ")
        named = groups_named(text)
        span = find(text)
        if span:
            if len(named - {group["zh"]}) + 1 <= max_groups:
                return dict(verify_status="verified_page", evidence=snippet(text, span),
                            publisher=publisher, page_title=page_title)
            return dict(verify_status="ambiguous",
                        evidence=f"page names {len(named)} groups: {''.join(sorted(named))[:60]}",
                        publisher=publisher, page_title=page_title)
        if len(named) > max_groups:      # list page that doesn't name ours -> don't trust the title either
            return dict(verify_status="unverified", evidence="", publisher=publisher, page_title=page_title)
    span = find(hit.title)
    if span:
        return dict(verify_status="verified_title", evidence=snippet(hit.title, span),
                    publisher=publisher, page_title=page_title or hit.title)
    return dict(verify_status="unverified", evidence="", publisher=publisher, page_title=page_title)


# --------------------------------------------------------------------------- download
def fetch_bytes(url: str, session: requests.Session, timeout: float, max_bytes: int,
                referer: str | None = None) -> bytes | None:
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        with session.get(url, headers=headers, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    return None
            return buf.getvalue()
    except (requests.RequestException, ValueError):
        return None


def to_jpeg(data: bytes, min_side: int, max_side: int = 0) -> tuple[bytes | None, str, tuple[int, int]]:
    """Validate and re-encode to RGB JPEG, shrunk so the longer side <= max_side (0 = keep).

    Returns (jpeg, reason, (original_w, original_h)). Shrinking keeps downloads small:
    SAM 3 works at segment.work_max_side anyway, so larger originals are never used.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                return None, "too_small", (w, h)
            if max_side and max(w, h) > max_side:
                s = max_side / max(w, h)
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            im.save(out, "JPEG", quality=92)
            return out.getvalue(), "", (w, h)
    except Exception:
        return None, "corrupt_or_not_image", (0, 0)


def examine(hit: Hit, group: dict, cfg: dict, session: requests.Session) -> dict:
    """Verify the page first (cheap reject), then download the image."""
    rec = dict(asdict(hit), group=group["code"])
    rec.pop("fallbacks", None)
    page = fetch_bytes(hit.page_url, session, cfg["page_timeout_s"], 3 * 1024 * 1024)
    rec.update(verify(hit, group, page, cfg["max_groups_named_on_page"]))
    rec["source_category"] = source_category(hit.page_url)
    rec["evidence_type"] = EVIDENCE_TYPE.get(rec["verify_status"], "")
    if rec["verify_status"].startswith("verified") and cfg.get("strict_evidence", True):
        ok, why = strict_evidence(group, rec.get("page_title", ""), rec.get("evidence", ""))
        if not ok:
            rec["verify_status"] = "ambiguous" if why == "multi_group_evidence" else "unverified"
            rec["strict_reason"] = why
    if not rec["verify_status"].startswith("verified"):
        rec["reason"] = rec.get("strict_reason") or rec["verify_status"]
        return rec
    reason = "download_failed"
    for i, url in enumerate((hit.image_url, *hit.fallbacks)):
        data = fetch_bytes(url, session, cfg["image_timeout_s"], int(cfg["max_image_mb"] * 1024 * 1024),
                           referer=hit.page_url if i == 0 else None)
        if data is None:
            continue
        jpeg, why, (w, h) = to_jpeg(data, cfg["min_side_px"], cfg.get("store_max_side", 0))
        if jpeg is None:
            reason = why if reason == "download_failed" or why == "too_small" else reason
            continue
        rec.update(width=w, height=h, downloaded_from="original" if i == 0 else url)
        rec["sha256"] = hashlib.sha256(jpeg).hexdigest()
        rec["_jpeg"] = jpeg
        return rec
    rec["reason"] = reason
    return rec


def shrink_file(path: Path, max_side: int) -> bool:
    """Re-save an already stored image with its longer side <= max_side. True if changed."""
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        if max(w, h) <= max_side:
            return False
        s = max_side / max(w, h)
        small = im.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    tmp = path.with_name(path.name + ".tmp")
    small.save(tmp, "JPEG", quality=92)
    tmp.replace(path)
    return True


# --------------------------------------------------------------------------- one group
RETRYABLE = ("download_failed", "corrupt_or_not_image")


def collect_group(group: dict, cfg: dict, out_dir: Path, urls: dict | None = None,
                  pause: float = 0.5, progress=print, retry_failed: bool = False) -> dict:
    """Search, verify and download one group into out_dir/<CODE>/ (resumable).

    Writes img/<ID>.jpg for saved images and appends every candidate (saved or not)
    to log.jsonl. Returns a small summary dict.
    """
    gdir = out_dir / group["code"]
    (gdir / "img").mkdir(parents=True, exist_ok=True)
    log_path = gdir / "log.jsonl"
    seen_urls, seen_sha, next_n = set(), set(), 1
    last_reason: dict[str, str | None] = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            seen_urls.add(r["image_url"])
            last_reason[r["image_url"]] = None if r.get("image_id") else r.get("reason")
            if r.get("sha256"):
                seen_sha.add(r["sha256"])
            if r.get("image_id"):
                next_n = max(next_n, int(r["image_id"][len(group["code"]):]) + 1)
    if retry_failed:                       # give failed downloads another go (with fallbacks)
        seen_urls -= {u for u, why in last_reason.items() if why in RETRYABLE}

    session = requests.Session()
    hits, dedup = [], set(seen_urls)
    for engine in cfg.get("engines", ["so360", "baidu", "bing"]):
        for q in group["queries"]:
            found = search(q, cfg["max_results_per_query"], session, engine, urls, pause)
            fresh = [h for h in found if h.image_url not in dedup]
            dedup.update(h.image_url for h in fresh)
            hits.extend(fresh)
            progress(f"[{group['code']}] {engine} '{q}': {len(found)} results, {len(fresh)} new")

    summary = {"group": group["code"], "candidates": len(hits), "saved": 0}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log_path, "a", encoding="utf-8") as log, \
            ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(examine, h, group, cfg, requests.Session()) for h in hits]
        for fut in as_completed(futures):
            rec = fut.result()
            rec["collected_at"] = stamp
            rec["license"] = LICENSE_UNKNOWN
            jpeg = rec.pop("_jpeg", None)
            if jpeg is not None and rec["sha256"] in seen_sha:
                rec["reason"], jpeg = "exact_duplicate", None
            if jpeg is not None:
                rec["image_id"] = f"{group['code']}{next_n:05d}"
                rec["file"] = f"{group['code']}/img/{rec['image_id']}.jpg"
                (gdir / "img" / f"{rec['image_id']}.jpg").write_bytes(jpeg)
                seen_sha.add(rec["sha256"])
                next_n += 1
                summary["saved"] += 1
            summary[rec.get("reason") or "saved_ok"] = summary.get(rec.get("reason") or "saved_ok", 0) + 1
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
    progress(f"[{group['code']}] done: {summary}")
    return summary

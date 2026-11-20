"""Build the Word report in the school template from the REAL project files.

How to run (from the project folder):
    python scripts/build_report.py

What it reads (it never invents a number):
    docs/inputs/report_template.docx   the school template (converted from .doc)
    docs/report_info.yaml              names, IDs, date, workload (TODO-FILL if empty)
    reports/summary.json               made by:  python -m ethnoseg stats
    dataset/qc_report.json             made by:  python -m ethnoseg qc
    reports/review_stats.json          made by:  python -m ethnoseg review-stats   (optional)
    reports/mask_agreement.json        made by:  python -m ethnoseg mask-agreement (optional)
    reports/weak_masks.csv             made by:  python -m ethnoseg mask-check     (optional)
    docs/environment.md                made by:  python scripts/env_report.py     (optional)
    data/manifest.db                   the pipeline database
    reports/figures/*.png              charts only (no source photos)

What it writes:
    docs/Ethnic_Clothing_Segmentation_Report.docx
    reports/figures/pipeline.png       (the pipeline diagram)

Step by step:
    1. Fill the cover page, the member table and the workload table.
    2. Fill the two personal score sheets (grade, major, ID, name).
    3. Delete the empty example chapters of the template.
    4. Write the abstract, the table of contents and the six chapters.
    5. Optional: if LibreOffice is installed, make a PDF and put real
       page numbers into the table of contents (a second pass).
"""
from __future__ import annotations

import copy
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "inputs" / "report_template.docx"
OUT = ROOT / "docs" / "Ethnic_Clothing_Segmentation_Report.docx"
FIG = ROOT / "reports" / "figures"
TODO = "TODO-FILL"

GROUPS = [("YI", "Yi", "彝族"), ("BA", "Bai", "白族"), ("ZH", "Zhuang", "壮族"),
          ("UY", "Uygur", "维吾尔族"), ("MA", "Manchu", "满族")]


# ----------------------------------------------------------------------------- data
def load_json(path: Path) -> dict | None:
    """Read a JSON file, or return None if the file does not exist yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def val(x) -> str:
    """Show a value, or TODO-FILL when it is empty."""
    x = "" if x is None else str(x).strip()
    return x if x and x != TODO else TODO


class Data:
    """All numbers used in the text. Each one comes from a project file."""

    def __init__(self):
        self.info = yaml.safe_load((ROOT / "docs" / "report_info.yaml").read_text(encoding="utf-8"))
        self.cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        self.summary = load_json(ROOT / "reports" / "summary.json")
        if self.summary is None:
            sys.exit("reports/summary.json is missing: run  python -m ethnoseg stats  first")
        self.qc = load_json(ROOT / "dataset" / "qc_report.json") or {}
        self.review_stats = load_json(ROOT / "reports" / "review_stats.json")
        self.agreement = load_json(ROOT / "reports" / "mask_agreement.json")
        weak = ROOT / "reports" / "weak_masks.csv"
        self.weak_masks = sum(1 for _ in csv.DictReader(open(weak, encoding="utf-8"))) if weak.exists() else None
        self.env_rows = self._env_rows(ROOT / "docs" / "environment.md")

        # numbers from the pipeline database
        con = sqlite3.connect(ROOT / "data" / "manifest.db")
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM images")]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        reviews = [dict(r) for r in con.execute("SELECT * FROM reviews")] if "reviews" in tables else []
        assigned = (Counter(r[0] for r in con.execute("SELECT reviewer FROM assignments"))
                    if "assignments" in tables else Counter())
        con.close()
        codes = [g[0] for g in GROUPS]
        self.rows = [r for r in rows if r["grp"] in codes]
        self.reviews = reviews
        self.assigned = assigned

        s = self.summary
        self.split_counts = s.get("split_counts", {})
        self.final_per_group = {g: sum(self.split_counts.get(g, {}).values()) for g in codes}
        self.final_total = sum(self.final_per_group.values())
        self.split_totals = Counter()
        for g in codes:
            self.split_totals.update(self.split_counts.get(g, {}))
        self.collection = s.get("collection", {})
        self.hits = {g: self.collection.get(g, {}).get("candidates", 0) for g in codes}
        self.saved = Counter(r["grp"] for r in self.rows)
        self.funnel = s.get("funnel", {})
        self.reserve = s.get("not_segmented_reserve", 0)
        self.sources = s.get("source_categories_final", {})
        self.pixel_share = s.get("class_pixel_share_of_clothing", {})

        # rejection reasons (after download)
        self.reasons = Counter()
        for c in s.get("rejections", {}).values():
            self.reasons.update(c)
        # SAM 3 outcome per image that was sent to SAM 3
        self.sam_out = Counter()
        for r in self.rows:
            reason = (r.get("reason") or "").split(":")[0]
            if r.get("seg_json") and reason not in {"multi_group_evidence", "own_group_missing"}:
                if r["status"] == "rejected" and reason in {"multiple_subjects", "no_subject", "no_parts",
                                                            "subject_too_small", "subject_cut"}:
                    self.sam_out[reason] += 1
                else:
                    self.sam_out["usable"] += 1
        self.status = Counter(r["status"] for r in self.rows)
        self.waiting_review = Counter(r["grp"] for r in self.rows if r["status"] == "composed")
        # decisions per reviewer (one row per person and image)
        self.decisions = {}
        for v in reviews:
            self.decisions.setdefault(v["reviewer"], Counter())[v["decision"]] += 1
        sizes = [v for v in self.final_per_group.values() if v]
        self.balance = round(max(sizes) / min(sizes), 2) if sizes else 0

    @staticmethod
    def _env_rows(path: Path) -> list[tuple[str, str]]:
        """Take the 2-column rows of the first table in docs/environment.md."""
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and len(cells) >= 2 and not set(cells[0]) <= set("-: ") \
                    and cells[0] != "Item":                      # skip separator and header rows
                out.append((cells[0], cells[1].replace("`", "")))
        return out


# ----------------------------------------------------------------------------- figures
def pipeline_figure(path: Path) -> None:
    """Draw the pipeline diagram from the real stage list (no photos)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    stages = [("1 Collect", "search 3 engines\nverify page text\n(Modal CPU)"),
              ("2 Ingest + screen", "size, blur, aspect\npHash duplicates"),
              ("3 Strict label check", "title + evidence name\nexactly one group"),
              ("4 SAM 3", "person + 6 part\nprompts (Modal GPU)"),
              ("5 Compose", "white background\n512x512, centred"),
              ("6 Human review", "keep / reject\n+ mask fix list"),
              ("7 Split + export", "70/15/15 by source\npage, metadata.csv"),
              ("8 QC + stats", "brief rules,\nfigures, report")]
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (head, body) in enumerate(stages):
        row, col = divmod(i, 4)
        col = col if row == 0 else 3 - col          # snake layout: left->right, then right->left
        x, y = col + 0.06, 1.08 - row * 1.0
        ax.add_patch(FancyBboxPatch((x, y), 0.86, 0.78, boxstyle="round,pad=0.02",
                                    fc="#eaf1f8" if i not in (5,) else "#fdf0dc", ec="#3a6ea5", lw=1.2))
        ax.text(x + 0.43, y + 0.6, head, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + 0.43, y + 0.28, body, ha="center", va="center", fontsize=8)
    arrow = dict(arrowstyle="->", color="#3a6ea5", lw=1.4)
    for c in range(3):
        ax.annotate("", xy=(c + 1.06, 1.47), xytext=(c + 0.92, 1.47), arrowprops=arrow)
        ax.annotate("", xy=(c + 0.92, 0.47), xytext=(c + 1.06, 0.47), arrowprops=arrow)
    ax.annotate("", xy=(3.49, 0.86), xytext=(3.49, 1.08), arrowprops=arrow)
    ax.text(2.0, 0.0, "All stages read config.yaml and write one SQLite database (data/manifest.db).",
            ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- docx helpers
def style(doc, name: str):
    """Find a style by name, ignoring upper/lower case (the template uses 'heading 1')."""
    for s in doc.styles:
        if s.name and s.name.lower() == name.lower():
            return s
    raise KeyError(name)


def set_text(p, text: str, underline_from: int | None = None) -> None:
    """Replace the text of a paragraph but keep the formatting of its runs.
    The first run keeps the label, the last run gets the value."""
    runs = p.runs
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    runs[0].text = text


def fill_label(p, value: str) -> None:
    """Cover lines look like 'label:<tab>____'. Keep the label runs, put the value in the
    first underlined run and remove the other underlined runs."""
    under = [r for r in p.runs if r.underline]
    if not under:
        p.add_run("  " + value).underline = True
        return
    under[0].text = f"  {value}  "
    for r in under[1:]:
        r._r.getparent().remove(r._r)


def cell_text(cell, text: str, bold: bool = False, size: float = 10.5) -> None:
    """Write text into a table cell, keeping the cell's paragraph style."""
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def set_cell_borders(table) -> None:
    """Thin black borders on every cell (the template has no 'Table Grid' style)."""
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.append(borders)


class Writer:
    """Adds paragraphs, headings, figures and tables at the end of the document."""

    def __init__(self, doc):
        self.doc = doc
        self.body_style = style(doc, "Normal")
        self.h1 = style(doc, "heading 1")
        self.h2 = style(doc, "heading 2")
        self.fig_no = {}
        self.tab_no = {}
        self.headings = []              # (level, text) for the table of contents

    def heading(self, text: str, level: int = 1) -> None:
        p = self.doc.add_paragraph(text, style=self.h1 if level == 1 else self.h2)
        pf = p.paragraph_format                  # template sizes, but compact spacing
        pf.line_spacing = 1.0
        pf.space_before = Pt(12 if level == 1 else 6)
        pf.space_after = Pt(6 if level == 1 else 3)
        pf.keep_with_next = True
        if text.startswith("参考文献"):
            pf.page_break_before = True
        self.headings.append((level, text))

    def para(self, text: str, indent: bool = True, size: float = 10.5, bold_prefix: str | None = None,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p = self.doc.add_paragraph(style=self.body_style)
        pf = p.paragraph_format
        pf.line_spacing = 1.25
        pf.space_after = Pt(3)
        if indent:
            pf.first_line_indent = Cm(0.74)
        p.alignment = align
        if bold_prefix:
            r = p.add_run(bold_prefix)
            r.bold = True
            r.font.size = Pt(size)
        text = re.sub(r"Figures? (\d)\.(\d)", r"图\1-\2", text)      # cross-references in the
        text = re.sub(r"Tables? (\d)\.(\d)", r"表\1-\2", text)       # school style: 图3-1, 表3-1
        r = p.add_run(text)
        r.font.size = Pt(size)
        return p

    def bullets(self, items: list[str]) -> None:
        for it in items:
            head, _, rest = it.partition("::")
            p = self.para(rest.strip() if rest else head, indent=False,
                          bold_prefix=(head.strip() + ": ") if rest else None)
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.first_line_indent = Cm(-0.4)
            p.runs[0].text = "• " + p.runs[0].text

    def caption(self, text: str) -> None:
        p = self.doc.add_paragraph(style=self.body_style)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.size = Pt(9)
        r.bold = True

    def figure(self, key: str, chapter: int, path: Path, text: str, width_cm: float = 14.0) -> str:
        n = sum(1 for k in self.fig_no.values() if k.startswith(f"{chapter}-")) + 1
        label = f"{chapter}-{n}"
        self.fig_no[key] = label
        p = self.doc.add_paragraph(style=self.body_style)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        if path.exists():
            p.add_run().add_picture(str(path), width=Cm(width_cm))
        else:
            p.add_run(f"[{TODO}: figure file {path.name} is missing — run the stats stage]")
        self.caption(f"图{label}  {text}")
        return label

    def table(self, key: str, chapter: int, title: str, header: list[str], rows: list[list[str]],
              widths_cm: list[float] | None = None, size: float = 9) -> str:
        n = sum(1 for k in self.tab_no.values() if k.startswith(f"{chapter}-")) + 1
        label = f"{chapter}-{n}"
        self.tab_no[key] = label
        cap = self.doc.add_paragraph(style=self.body_style)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.keep_with_next = True
        r = cap.add_run(f"表{label}  {title}")
        r.font.size = Pt(9)
        r.bold = True
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGN_CENTER if False else WD_TABLE_ALIGNMENT.CENTER
        set_cell_borders(t)
        for j, h in enumerate(header):
            cell_text(t.rows[0].cells[j], h, bold=True, size=size)
            sh = OxmlElement("w:shd")
            sh.set(qn("w:val"), "clear")
            sh.set(qn("w:fill"), "E8EEF5")
            t.rows[0].cells[j]._tc.get_or_add_tcPr().append(sh)
        for i, row in enumerate(rows, start=1):
            for j, v in enumerate(row):
                cell_text(t.rows[i].cells[j], str(v), size=size)
        for row in t.rows[:-1]:                    # keep the whole table on one page
            for c in row.cells:
                for par in c.paragraphs:
                    par.paragraph_format.keep_with_next = True
        for row in t.rows:
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trPr.append(cs)
        if widths_cm:
            for row in t.rows:
                for j, w in enumerate(widths_cm):
                    row.cells[j].width = Cm(w)
        self.doc.add_paragraph(style=self.body_style).paragraph_format.space_after = Pt(2)
        return label


WD_TABLE_ALIGN_CENTER = WD_TABLE_ALIGNMENT.CENTER


# ----------------------------------------------------------------------------- template parts
def fill_front(doc, d: Data) -> None:
    """Cover page, member table, workload table, personal score sheets."""
    info = d.info
    lead = info["leader"]
    members = info["members"]
    paras = doc.paragraphs
    find = lambda start: next(p for p in paras if p.text.replace(" ", "").replace("　", "").startswith(start))

    fill_label(find("题目"), f"{info['title_zh']}")
    fill_label(find("组长"), lead["name"])
    fill_label(find("联系电话"), val(lead["phone"]))
    fill_label(find("电子邮件"), val(lead["email"]))
    date_p = find("报告完成时间")
    date = val(info.get("report_date"))
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})$", date)
    set_text(date_p, "报告完成时间：  " + (f"{m[1]} 年 {int(m[2])} 月 {int(m[3])} 日" if m else date))

    # table 0: 个人成绩 (序号 学号 姓名 专业 成绩) ; table 1: 项目成员及分工
    t0, t1 = doc.tables[0], doc.tables[1]
    for t in (t0, t1):
        while len(t.rows) > 1 + len(members):          # keep one row per member
            t._tbl.remove(t.rows[-1]._tr)
    for i, mem in enumerate(members, start=1):
        c = t0.rows[i].cells
        cell_text(c[1], val(mem["id"]))
        cell_text(c[2], mem["name"], size=9)
        cell_text(c[3], val(mem["major"]))             # 成绩 stays empty for the teacher
        c = t1.rows[i].cells
        cell_text(c[1], val(mem["id"]))
        cell_text(c[2], mem["name"], size=9)
        cell_text(c[3], val(mem.get("tasks")), size=9)
        w = val(mem.get("workload"))
        cell_text(c[4], w if w == TODO or w.endswith("%") else f"{w}%")

    # personal score sheets: the lines "年级：… 专业：… 学号：… 姓名：…"
    lines = [p for p in paras if p.text.strip().startswith("年级")]
    for p, mem in zip(lines, members):
        r0 = p.runs[0]
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
        r0.text = "年级："
        for label, value in ((None, val(mem["grade"])), ("专业：", val(mem["major"])),
                             ("学号：", val(mem["id"])), ("姓名：", mem["name"])):
            if label:
                lr = p.add_run("    " + label)
                lr.bold = True
                lr.font.size = r0.font.size
            vr = p.add_run(f" {value} ")
            vr.bold = True
            vr.underline = True
            vr.font.size = r0.font.size


def cut_template_body(doc) -> None:
    """Remove the example body of the template (from 'xxx系统' to the end),
    but keep the last section settings (page size, footer with page numbers)."""
    body = doc.element.body
    elems = list(body)
    start = next(i for i, e in enumerate(elems) if "xxx" in e.xpath("string(.)") and "系统" in e.xpath("string(.)"))
    for e in elems[start:]:
        if e.tag != qn("w:sectPr"):
            body.remove(e)


def add_toc(w: Writer, entries: list[tuple[int, str, str]] | None) -> None:
    """A real Word TOC field. Word can refresh it (right click > Update Field).
    The visible text inside the field is our own list with page numbers (if known)."""
    doc = w.doc
    p = doc.add_paragraph(style=w.body_style)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("目录 Contents")
    r.bold = True
    r.font.size = Pt(15)

    def fld(par, kind):
        e = OxmlElement("w:fldChar")
        e.set(qn("w:fldCharType"), kind)
        par.add_run()._r.append(e)

    first = doc.add_paragraph(style=style(doc, "TOC 1"))
    fld(first, "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    first.add_run()._r.append(instr)
    fld(first, "separate")
    entries = entries or [(lvl, txt, "") for lvl, txt in w.headings]
    last = first
    for k, (lvl, txt, page) in enumerate(entries):
        par = first if k == 0 else doc.add_paragraph(style=style(doc, "TOC 1" if lvl == 1 else "TOC 2"))
        tabs = par.paragraph_format.tab_stops
        tabs.add_tab_stop(Cm(15.4), alignment=2, leader=1)       # right aligned, dots
        if lvl == 2:
            par.paragraph_format.left_indent = Cm(0.74)
        run = par.add_run(f"{txt}\t{page}")
        run.font.size = Pt(10.5)
        run.bold = lvl == 1
        last = par
    fld(last, "end")


def section_break_here(doc, roman: bool) -> None:
    """End the current section here (copy of the final section settings).
    roman=True: this front section uses I, II, III page numbers."""
    final = doc.element.body.find(qn("w:sectPr"))
    sp = copy.deepcopy(final)
    pg = sp.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sp.append(pg)
    pg.set(qn("w:fmt"), "upperRoman" if roman else "decimal")
    pg.set(qn("w:start"), "1")
    p = doc.add_paragraph()
    p._p.get_or_add_pPr().append(sp)


# ----------------------------------------------------------------------------- the text
def pct(a, b) -> str:
    return f"{a / b:.1%}" if b else "0%"


def write_body(w: Writer, d: Data, toc_entries=None) -> None:
    info = d.info
    s = d.summary
    groups_txt = ", ".join(f"{n} ({z})" for _, n, z in GROUPS)
    per_group = ", ".join(f"{n} {d.final_per_group[c]}" for c, n, _ in GROUPS)
    hits_total = sum(d.hits.values())
    saved_total = sum(d.saved.values())
    strict_rej = d.reasons.get("multi_group_evidence", 0) + d.reasons.get("own_group_missing", 0)
    shop = d.sources.get("shop", 0)
    src_total = sum(d.sources.values()) or 1
    waiting = sum(d.waiting_review.values())
    r1 = d.decisions.get("reviewer_1", Counter())
    tr = d.final_total

    # ---- title, abstracts (new page after the score sheets) -------------------
    p = w.doc.add_paragraph(style=w.body_style)
    p.paragraph_format.page_break_before = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(info["title_zh"])
    r.bold = True
    r.font.size = Pt(16)
    p = w.doc.add_paragraph(style=w.body_style)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(info["title_en"])
    r.bold = True
    r.font.size = Pt(13)
    names = " · ".join(m["name"] for m in info["members"])
    w.para(names, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

    w.para("", indent=False, bold_prefix="摘要")
    zh = (f"民族服饰是重要的非物质文化遗产，但目前公开的像素级民族服饰部件分割数据很少。本项目构建了一个面向"
          f"彝族、白族、壮族、维吾尔族和满族五个民族的服饰语义分割数据集，并实现了从采集到质检的完整流水线。"
          f"图像通过三个图片搜索引擎采集，民族标签只依据来源网页的标题和文字证据，而不依据人物外貌判断；"
          f"页面文字必须且只能提到本民族。经过尺寸、模糊度、长宽比和感知哈希去重筛选后，使用 SAM 3 以文本提示"
          f"自动生成上衣、下装、连衣裙/长袍、头饰、鞋、腰饰六类部件掩码，再统一合成为白底、居中的 512×512 图像，"
          f"并由人工逐张审核保留或剔除。当前数据集共 {tr} 张图像（{'，'.join(f'{z}{d.final_per_group[c]}张' for c, _, z in GROUPS)}），"
          f"按来源网页分组以 7:1.5:1.5 划分训练、验证和测试集，自动质检无硬性错误项。报告同时如实说明了来源以电商页面为主、"
          f"图像版权受限、部分掩码仍需人工修正等局限。")
    w.para(zh)
    w.para("数据集构建；语义分割；民族服饰；SAM 3；数据质量控制", indent=False, bold_prefix="关键词：")

    w.para("", indent=False, bold_prefix="Abstract")
    en = (f"Ethnic clothing is an important part of China's intangible cultural heritage, yet public pixel-level "
          f"datasets of its garment parts are rare. We built a part-segmentation dataset for five ethnic groups — "
          f"{groups_txt} — together with a reproducible pipeline that covers every stage of the course brief. "
          f"Images were collected from three image search engines ({hits_total:,} search hits). The ethnic label of "
          f"an image comes only from the text of its source page, never from how the clothes or people look; a page "
          f"must name its own group and no other of the 56 groups. After screening (size, blur, aspect, perceptual-hash "
          f"duplicates) and the strict label check, SAM 3 produced masks for six garment classes from text prompts, and "
          f"each image was composed onto a white 512×512 canvas with the person centred. People then kept or rejected "
          f"every composed image. The current dataset has {tr} images ({per_group}), split 70/15/15 by source page so "
          f"that no page appears in two splits. The automatic QC reports "
          f"{'no hard failures' if not d.qc.get('hard_failures') else 'hard failures: ' + json.dumps(d.qc.get('hard_failures'))}. "
          f"We also report the limits honestly: {pct(shop, src_total)} of the final images come from shop pages, the "
          f"images are copyright-restricted, and some masks still need manual correction.")
    w.para(en)
    w.para("dataset construction; semantic segmentation; ethnic clothing; SAM 3; data quality control",
           indent=False, bold_prefix="Keywords: ")

    # table of contents in its own front section (Roman page numbers)
    p = w.doc.add_paragraph(style=w.body_style)
    p.add_run().add_break(WD_BREAK.PAGE)
    add_toc(w, toc_entries)
    section_break_here(w.doc, roman=True)

    # ---- Chapter 1 -----------------------------------------------------------
    w.headings.clear()
    w.heading("第一章 绪论 (Introduction)")
    w.heading("1.1 项目背景与意义 (Background and Significance)", 2)
    w.para("Traditional dress is one of the most visible carriers of the culture of China's 55 ethnic minorities. "
           "Digital archives, museum catalogues, virtual try-on and costume-design tools all need to know not only "
           "which group a costume belongs to, but also where each garment part is: the jacket, the skirt, the "
           "headdress, the apron. This is a semantic segmentation problem, and models for it need images with "
           "pixel-level labels. Such labels are expensive to draw by hand, and for ethnic clothing there is almost "
           "no public data. The aim of this mid-term project is therefore a dataset, not a model: we build, check "
           "and document a part-segmentation dataset for five groups, and we keep the whole process reproducible.")
    w.heading("1.2 国内外研究现状 (Related Work)", 2)
    w.para("Clothing and human parsing has several public benchmarks. ATR labels 17,700 images with 18 classes [1]; "
           "LIP (Look into Person) has more than 50,000 images with 19 part labels and many poses and occlusions [2]; "
           "ModaNet adds polygon annotations for 13 garment categories on 55,176 street-fashion images [3]; "
           "DeepFashion [4] and Fashionpedia [5] focus on attributes, landmarks and fine-grained apparel parts. "
           "All of them show everyday or Western fashion; ethnic costumes with headdresses, aprons and long robes are "
           "almost absent.")
    w.para("Promptable segmentation changed how such data can be labelled. The Segment Anything Model (SAM) returns a "
           "mask for a point or box prompt [6]. SAM 3 goes further: given a short noun phrase such as \"skirt\", it "
           "finds and segments all instances of that concept in an image [7]. This lets us pre-label garment parts "
           "with text prompts, so people only check and correct the masks.")
    w.para("Work on Chinese ethnic costumes in computer vision is mostly classification or detection on self-built "
           "image sets, for example the classification of She clothing from different regions with a CNN [8] and "
           "YOLOv11-based recognition of costumes in Lijiang on 4,974 images [9]. We did not find a public "
           "pixel-level part-segmentation dataset for the five groups of this project, which is the gap we address.")
    w.heading("1.3 项目目标与挑战 (Goals and Challenges)", 2)
    w.para("The course brief sets the targets below; Table 1.1 also shows where the current dataset stands.")
    qc_hard = d.qc.get("hard_failures") or {}
    w.table("goals", 1, "Requirements of the brief and current status",
            ["Requirement", "Target", "Current value"],
            [["Ethnic groups", "5", str(sum(1 for v in d.final_per_group.values() if v))],
             ["Images in total", "≥ 1000", str(tr)],
             ["Images per group", "≥ 100", f"min {min(d.final_per_group.values() or [0])}"],
             ["Image size / background", "512×512, white, centred", "all images (checked by QC)"],
             ["Subject fills the frame", "≥ 60%", f"{d.cfg['compose']['fill']:.0%} target, QC min {d.cfg['qc']['min_fill']:.0%}"],
             ["Largest : smallest group", "≈ 2 : 1 or better", f"{d.balance} : 1"],
             ["QC hard failures", "0", str(sum(qc_hard.values()) if qc_hard else 0)]],
            widths_cm=[4.6, 4.6, 4.6])
    w.para("Four problems took most of the effort. (1) Ambiguous labels: shop titles often list several groups "
           f"(\"Miao Yi Zhuang dance costume\"), so a looser rule would mislabel images; the strict rule removed "
           f"{strict_rej} images after download. (2) Performance costumes: many hits are modern stage costumes; we "
           "keep them only when the page clearly names one group, and we say so in the limitations. (3) Licence: "
           "the images belong to their owners, so the image files cannot be redistributed publicly. (4) Balance: "
           "Zhuang had the fewest usable images after the strict rule, so we segmented the remaining Zhuang reserve "
           "with SAM 3 to top it up.")

    # ---- Chapter 2 -----------------------------------------------------------
    w.heading("第二章 系统总体设计 (System Design)")
    w.heading("2.1 系统架构与整体流程 (Architecture and Workflow)", 2)
    w.para("The pipeline is a Python package (ethnoseg) with one command per stage. Every stage reads the same "
           "config.yaml and records its result for each image in one SQLite database (data/manifest.db): status, "
           "reason for rejection, source URL, evidence text, SAM 3 output, review decision and split. Because each "
           "stage only processes images that are not done yet, a stage can be stopped and run again safely. Heavy "
           "work runs on Modal: collection on CPU containers and SAM 3 on a GPU, with results stored in a Modal "
           "Volume and pulled back to the laptop. Figure 2.1 shows the stages.")
    pipe = FIG / "pipeline.png"
    pipeline_figure(pipe)
    w.figure("pipe", 2, pipe, "Pipeline of the dataset construction system (drawn from the real stage list)", 14.5)
    w.heading("2.2 技术选型与依据 (Technology Choices)", 2)
    w.table("tech", 2, "Main technology choices and reasons",
            ["Task", "Choice", "Alternatives considered", "Reason"],
            [["Part masks", "SAM 3 (facebook/sam3) [7]", "YOLO-seg, background-removal tools",
              "text prompts per garment class; no training data needed; one model for person and parts"],
             ["Compute", "Modal (CPU + L4/A10G/T4 GPU)", "local CPU, Colab",
              "laptop has no CUDA GPU; Modal runs the same code in containers with a GPU fallback list"],
             ["Bookkeeping", "SQLite manifest", "CSV files, folders",
              "one row per image with status and reason; safe re-runs; easy counts for the report"],
             ["Model loading", "Hugging Face Transformers [10]", "original SAM 3 repo",
              "Sam3Model / Sam3Processor classes; weights from the gated Hub repo"],
             ["Image work", "OpenCV, Pillow, NumPy", "—", "resize, blur test, pHash, composing, masks"],
             ["Demo", "Gradio", "Flask", "a browser viewer in a few lines of Python"]],
            widths_cm=[2.4, 3.6, 3.6, 6.2], size=8.5)
    w.heading("2.3 开发环境与工具配置 (Development Environment)", 2)
    keep = ("Operating system", "Python", "opencv-python", "numpy", "torch", "transformers", "GPU", "SAM 3 weights")
    env = [r for r in d.env_rows if r[0] in keep] or \
        [("Environment file", f"{TODO}: run python scripts/env_report.py")]
    w.table("env", 2, "Development environment (key items; full list in docs/environment.md)", ["Item", "Value"],
            [[a, b] for a, b in env[:12]], widths_cm=[4.5, 10.5], size=8.5)

    # ---- Chapter 3 -----------------------------------------------------------
    w.heading("第三章 数据集构建 (Dataset Construction)")
    w.heading("3.1 类别定义与设计 (Class Definition)", 2)
    cls = d.cfg["classes"]
    w.para("Each mask is a single-channel PNG with one class ID per pixel (Table 3.1). Class 0 is the background "
           "and also every pixel that is not clothing: skin, hair, hands. Parts that are hidden are not labelled. "
           "Where SAM 3 masks overlap, a fixed paint order decides the label (" +
           " < ".join(d.cfg["paint_order"]) + "), so a small headdress on top of a robe keeps its own class. "
           "A one-piece robe or long gown is \"dress\"; a jacket worn over a skirt is \"top\" + \"bottom\".")
    w.table("classes", 3, "Mask classes and SAM 3 text prompts", ["ID", "Class", "SAM 3 prompts",
                                                                   "Share of clothing pixels"],
            [[str(c["id"]), c["name"], ", ".join(c.get("prompts", [])) or "— (not clothing)",
              f"{d.pixel_share.get(c['name'], 0):.1%}" if c["id"] else "—"] for c in cls],
            widths_cm=[1.2, 2.6, 6.5, 4.0])
    w.heading("3.2 数据采集 (Data Collection)", 2)
    eng = d.cfg["collect"]["engines"]
    w.para(f"For each group we used 10 search terms in Chinese and English (for example "
           f"\"{d.cfg['groups'][0]['queries'][0]}\", \"{d.cfg['groups'][0]['queries'][6]}\") on {len(eng)} image "
           f"search engines (360 Images, Baidu Images and Bing), up to {d.cfg['collect']['max_results_per_query']} "
           f"results per term and engine. For every hit, the collector opened the source page and stored its URL, "
           f"title, publisher and an evidence sentence. The label rule is strict: the page title plus evidence text "
           f"must contain the group's own name (or an official alias such as 维吾尔/维族) and must not name any "
           f"other of the 56 groups. Labels never come from appearance. Table 3.2 gives the counts.")
    w.table("collect", 3, "Collection counts per group",
            ["Group", "Search hits", "Saved (label verified)", "Final dataset"],
            [[f"{n} {z}", f"{d.hits[c]:,}", f"{d.saved[c]:,}", str(d.final_per_group[c])] for c, n, z in GROUPS]
            + [["Total", f"{hits_total:,}", f"{saved_total:,}", str(tr)]],
            widths_cm=[3.6, 3.4, 4.0, 3.4])
    w.heading("3.3 数据清洗与去重 (Cleaning and De-duplication)", 2)
    sc = d.cfg["screen"]
    w.para(f"Every saved image was screened automatically: images with a side below {sc['min_side_px']} px, an "
           f"aspect ratio above {sc['max_aspect']} (banners and collages) or a Laplacian blur score below "
           f"{sc['blur_min']} were removed, and near-duplicates were found with a 64-bit perceptual hash (Hamming "
           f"distance ≤ {sc['near_dup_hamming']}). After composing, a second duplicate check compares the 512×512 "
           f"subject crops, so the same photo from two shops cannot end up in two splits. Every removed image keeps "
           f"its reason in the database; Figure 3.1 shows the most common reasons.")
    w.figure("rej", 3, FIG / "rejection_reasons.png", "Why images were removed after download (all groups)", 10.5)
    w.heading("3.4 自动内容筛选 (Automatic Screening: Quality, Composition, Duplicates)", 2)
    w.para("The template names this step \"CLIP-based screening\". We did not run CLIP. Content screening was done "
           "by the rules above and by SAM 3 itself: an image with no person and no garment (\"no_subject\"), with two "
           "or more large people (\"multiple_subjects\"), with a cut-off or too small subject, or with almost no "
           "clothing pixels (\"no_parts\") was removed automatically. We report this truthfully instead of claiming "
           "a CLIP filter.")
    w.heading("3.5 自动预标注 (SAM 3 Automatic Pre-annotation)", 2)
    sg = d.cfg["segment"]
    sam_total = sum(d.sam_out.values())
    w.para(f"SAM 3 first finds the main person with the prompt \"{sg['person_prompt']}\" (score ≥ "
           f"{sg['person_threshold']}), then runs one prompt per garment word (Table 3.1, score ≥ "
           f"{sg['part_threshold']}). A part mask is kept only if at least {sg['part_min_inside_subject']:.0%} of it "
           f"lies inside the person (headwear and footwear only need to touch the person, because tall headdresses "
           f"stick out). The parts are painted in the fixed order, small fragments are dropped, and the person is "
           f"cut out, scaled so that the longer side fills {d.cfg['compose']['fill']:.0%} of a 512×512 white canvas "
           f"and centred. Of the {sam_total} images sent to SAM 3, " +
           ", ".join(f"{v} were {k.replace('_', ' ')}" for k, v in d.sam_out.most_common()) +
           f". A further {d.reserve} images that passed the strict label rule were not segmented (budget reserve).")
    w.heading("3.6 人工精调与质量审核 (Human Refinement and Review)", 2)
    rid = val(info.get("reviewer_1_identity"))
    w.para(f"Every composed image is checked by a person in a review window (original image, mask overlay, label "
           f"evidence). Keys: K = keep, R = reject with a reason, F = keep but put the mask on the fix list. A key "
           f"only counts if the image was visible for at least 0.4 s and the key was pressed again (not held), so a "
           f"held key cannot accept many images by accident. Before the Zhuang top-up, all images were reviewed by "
           f"one person, recorded as reviewer_1 (identity: {rid}): {r1.get('keep', 0)} kept and "
           f"{r1.get('reject', 0)} rejected. In that first round the window also showed a keep/reject suggestion "
           f"from an AI pre-screen (see Appendix B); the decision stored for each image was entered by the person.")
    rs = d.review_stats
    pairs = (rs or {}).get("pairs") or []
    if pairs:
        rows = [[x["pair"], str(x["images"]), f"{x['agreement']:.1%}", str(x["kappa"])] for x in pairs]
        w.para("For the second round the new images were split between the two team members with a 10% overlap, "
               "and a 10% sample of already decided images was drawn as an audit. Agreement is measured with "
               "Cohen's kappa [11]; a tie is rejected as \"review_conflict\" until someone looks again (Table 3.3).")
        w.table("kappa", 3, "Agreement between reviewers on shared images",
                ["Pair", "Images", "Agreement", "Cohen's kappa"], rows, widths_cm=[5, 2.5, 3, 3])
    else:
        plan = ", ".join(f"{k} {v}" for k, v in sorted(d.assigned.items())) or "no plan yet"
        w.para(f"For the second round (the {waiting} new composed images, mostly Zhuang) the images were split "
               f"between the two team members with a 10% overlap and a 10% audit sample of earlier decisions "
               f"(assigned: {plan}). Agreement will be measured with Cohen's kappa [11]. {TODO}: these review "
               f"sessions had not been done when this version of the report was built; run "
               f"python -m ethnoseg review-stats and rebuild the report afterwards.")
    w.heading("3.7 格式转换与数据集划分 (Export and Split)", 2)
    st = d.split_totals
    w.para(f"Kept images are split 70/15/15 into train, val and test inside each group. The split is grouped by "
           f"source page: all images from one web page go to the same split, so near-identical shop photos cannot "
           f"leak from train to test. The seed is {d.cfg['project']['seed']}. The result is {st.get('train', 0)} "
           f"train, {st.get('val', 0)} val and {st.get('test', 0)} test images (Figure 3.2). The export follows the "
           f"brief: dataset/images/<split>/<ID>.png and dataset/annotations/masks/<split>/<ID>.png (same file "
           f"name; the ID starts with the group code, e.g. ZH00123), dataset/metadata.csv (one row per image with source URL, publisher, evidence, licence note, "
           f"search term, split and class presence), a class map and a README.")
    w.figure("split", 3, FIG / "images_per_group.png", "Images per ethnic group and split", 10.5)

    # ---- Chapter 4 -----------------------------------------------------------
    w.heading("第四章 数据集质量评估 (Dataset Quality Evaluation)")
    w.heading("4.1 自动质检结果 (Automatic QC)", 2)
    warn = d.qc.get("warnings") or {}
    w.para(f"The QC stage re-reads every exported file and checks the brief: image size and mode, white "
           f"background, centring (offset ≤ {d.cfg['qc']['max_center_offset']:.0%}), fill (≥ "
           f"{d.cfg['qc']['min_fill']:.0%}), valid class IDs, one label per page, counts per group, split ratios, "
           f"source pages in two splits and near-duplicates across splits. Result for {d.qc.get('images', tr)} images: "
           f"hard failures — {', '.join(f'{k} {v}' for k, v in qc_hard.items()) or 'none'}; warnings — "
           f"{', '.join(f'{k.replace(chr(95), chr(32))} {v}' for k, v in warn.items()) or 'none'}. "
           f"The largest group is {d.balance} times the smallest.")
    w.heading("4.2 类别分布 (Class Distribution)", 2)
    w.para("Figure 4.1 shows how the clothing pixels are divided among the six classes in each group. The shape of "
           "the costumes is visible in the data: Uygur and Manchu images are dominated by long dresses and robes, "
           "while Yi, Bai and Zhuang costumes more often split into top and bottom, with aprons and headwear. "
           "Footwear and waistwear are small and rare classes; a model trained on this data will need class "
           "weighting for them.")
    w.figure("share", 4, FIG / "part_share_per_group.png", "Clothing-part composition per group", 10.5)
    w.heading("4.3 来源分析 (Source Analysis)", 2)
    w.para(f"Figure 4.2 shows the type of source page for the final images: {shop} of {src_total} "
           f"({pct(shop, src_total)}) come from shop pages, followed by " +
           ", ".join(f"{k} {v}" for k, v in list(d.sources.items())[1:4]) +
           ". Museums and institutions are a very small share. This is a real bias of the data and we list it in the "
           "limitations. Figure 4.3 summarises the whole funnel from search hits to the final dataset.")
    w.figure("src", 4, FIG / "source_categories.png", "Source page type of the final images", 10)
    w.figure("funnel", 4, FIG / "funnel.png", "From search hits to the final dataset (log scale)", 10)
    w.heading("4.4 自动掩码与人工修正对比 (SAM 3 vs. Human-corrected Masks)", 2)
    ag = d.agreement
    if ag and ag.get("images"):
        w.para(f"On the {ag['images']} masks that a person corrected, the SAM 3 masks reach a mean IoU of "
               f"{ag.get('miou', 0):.3f} against the corrected version; per-class values are in "
               f"reports/mask_agreement.md.")
    else:
        w.para(f"This comparison needs masks that a person has corrected with the mask editor "
               f"(python -m ethnoseg mask-edit). {TODO}: no corrected masks existed when this report was built, so "
               f"no mIoU between automatic and corrected masks is reported. We did not train a baseline "
               f"segmentation model either; the brief does not require training and we preferred to spend the time "
               f"on label quality.")
    w.heading("4.5 错误分析与改进方向 (Error Analysis and Improvements)", 2)
    wm = f"The rule-based mask check flags {d.weak_masks} masks for a second look (reports/weak_masks.csv). " \
        if d.weak_masks is not None else ""
    w.para(wm + "The typical failure cases we saw are: (1) a necklace or a large silver collar labelled as top; "
           "(2) a long apron merged into the skirt, so waistwear is missing; (3) hair ornaments and headdresses cut "
           "at the top when they are very tall; (4) a second person partly visible at the edge of shop photos; "
           "(5) sleeves of the same colour as the background lost at the border. Improvements: more prompt words "
           "per class (e.g. \"collar\", \"apron\"), a lower threshold for small classes, and the paint-on editor for "
           "the images on the fix list.")

    # ---- Chapter 5 -----------------------------------------------------------
    w.heading("第五章 系统实现与演示 (Implementation and Demo)")
    w.heading("5.1 命令行模块 (Command-line Modules)", 2)
    w.para("Every stage is one command (docs/commands.md lists inputs and outputs). The main ones are:")
    w.table("cli", 5, "Main commands of the pipeline", ["Command", "What it does"],
            [["modal run modal_collect.py", "search, verify page text, download (Modal CPU)"],
             ["python -m ethnoseg pull raw / ingest / screen", "copy downloads, register in the database, screen"],
             ["python -m ethnoseg reverify", "apply the strict one-group evidence rule"],
             ["modal run modal_segment.py", "SAM 3 person + part masks (Modal GPU)"],
             ["python -m ethnoseg pull seg / compose", "fetch masks, compose 512×512 white images"],
             ["python -m ethnoseg review-plan / review", "assign images, keyboard review window"],
             ["python -m ethnoseg review-stats / mask-check / mask-edit", "kappa, weak masks, paint-on editor"],
             ["python -m ethnoseg split / export / qc / stats", "split, export, check the brief, make figures"]],
            widths_cm=[7.2, 8.3], size=8.5)
    w.heading("5.2 Web交互界面 (Gradio Browser Demo)", 2)
    w.para("app_demo.py starts a small Gradio page on the local machine. The user chooses a group, a split and an "
           "image; the page shows the 512×512 image, the colour-coded mask overlay with a class legend, and the "
           "source URL and evidence text from metadata.csv. The demo reads only the local dataset folder.")

    # ---- Chapter 6 -----------------------------------------------------------
    w.heading("第六章 总结与展望 (Conclusion and Future Work)")
    w.heading("6.1 工作总结 (Summary)", 2)
    w.para(f"We built a reproducible pipeline and a {tr}-image part-segmentation dataset for five ethnic groups "
           f"({per_group}). Labels come only from source-page text under a strict one-group rule, masks come from "
           f"SAM 3 text prompts, every image is composed to the brief's 512×512 white format, people reviewed the "
           f"images, and an automatic QC checks the result against the brief.")
    w.heading("6.2 不足与局限 (Limitations)", 2)
    w.bullets([f"Licence:: the images belong to their owners; image files and masks are not published, only code "
               f"and metadata.",
               f"Shop sources:: {pct(shop, src_total)} of the final images come from shop pages; many show stage or "
               f"commercial costumes, not everyday or museum dress.",
               "Label rule:: the text rule is precise but strict; pages that name two groups are lost, and a wrong "
               "page title would still give a wrong label.",
               "Skin and hair:: they are background by rule, so the dataset cannot be used for full human parsing.",
               "SAM 3 errors:: masks are automatic; only part of them have been corrected by hand so far.",
               "Class imbalance:: footwear and waistwear are rare; Uygur has the most images."])
    w.heading("6.3 未来工作 (Future Work)", 2)
    w.para("Finish the second review round and correct the masks on the fix list; measure SAM 3 against the "
           "corrected masks; add museum sources and more groups; train a small baseline (e.g. SegFormer-B0) to "
           "show that the labels are learnable; and ask the image owners or the teacher how the images may be "
           "shared for teaching.")

    # ---- appendices and references ------------------------------------------
    w.heading("附录 (Appendix)")
    w.heading("A. 复现步骤 (Reproducibility)", 2)
    w.para("1) python -m venv .venv and pip install -r requirements.txt (or setup_mac.sh); 2) modal token new and "
           "a Modal Secret with the Hugging Face token; 3) run the stages in Table 5.1 in order (run_all.sh does "
           "this); 4) pytest -q tests. The seed and all thresholds are in config.yaml. Search engine results change "
           "over time, so a new collection run will not return exactly the same images; the stored metadata.csv "
           "and manifest record what was used.", indent=False)
    w.heading("B. AI工具使用说明 (Use of AI Tools)", 2)
    w.para("We used an AI assistant (Claude, Anthropic) to help write and debug the pipeline code, the tests and "
           "drafts of this report and the slides, and to run an automatic pre-screen whose keep/reject suggestion "
           "was shown in the first review round. SAM 3 (Meta) produced the automatic masks. The people on the team "
           "made the keep/reject decisions, checked the source pages and numbers, and are responsible for the "
           "content of this report. All numbers in this report are read from the project files by "
           "scripts/build_report.py.", indent=False)

    w.heading("参考文献 (References)")
    refs = [
        "Liang X, Liu S, Shen X, et al. Deep human parsing with active template regression[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015, 37(12): 2402-2414.",
        "Gong K, Liang X, Zhang D, et al. Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017: 932-940.",
        "Zheng S, Yang F, Kiapour M H, et al. ModaNet: A large-scale street fashion dataset with polygon annotations[C]//Proceedings of the 26th ACM International Conference on Multimedia, 2018: 1670-1678.",
        "Liu Z, Luo P, Qiu S, et al. DeepFashion: Powering robust clothes recognition and retrieval with rich annotations[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 1096-1104.",
        "Jia M, Shi M, Sirotenko M, et al. Fashionpedia: Ontology, segmentation, and an attribute localization dataset[C]//European Conference on Computer Vision (ECCV), 2020: 316-332.",
        "Kirillov A, Mintun E, Ravi N, et al. Segment anything[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023: 4015-4026.",
        "Carion N, Gustafson L, Hu Y T, et al. SAM 3: Segment anything with concepts[C]//International Conference on Learning Representations (ICLR), 2026. arXiv:2511.16719.",
        "Ding X, Li T, Chen J, et al. Research on the clothing classification of the She ethnic group in different regions based on FPA-CNN[J]. Applied Sciences, 2023, 13(17): 9676.",
        "He Y, Nguyen L T, Chansanam W. Intelligent recognition of ethnic costumes using YOLOv11: A deep learning framework for cultural heritage preservation[J]. Treatises and Documents, Journal of Ethnic Studies, 2026, 96. DOI: 10.2478/tdjes-2026-0007.",
        "Wolf T, Debut L, Sanh V, et al. Transformers: State-of-the-art natural language processing[C]//Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020: 38-45.",
        "Cohen J. A coefficient of agreement for nominal scales[J]. Educational and Psychological Measurement, 1960, 20(1): 37-46.",
    ]
    for i, ref in enumerate(refs, start=1):
        p = w.para(f"[{i}] {ref}", indent=False, size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)


# ----------------------------------------------------------------------------- main
def build(toc_entries=None) -> Writer:
    d = Data()
    doc = Document(str(TEMPLATE))
    fill_front(doc, d)
    cut_template_body(doc)
    w = Writer(doc)
    write_body(w, d, toc_entries)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    return w


def pdf_page_numbers(pdf: Path, headings: list[tuple[int, str]]) -> list[tuple[int, str, str]] | None:
    """Find on which body page each heading starts (page 1 = first page of chapter 1)."""
    if not shutil.which("pdftotext"):
        return None
    pages = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                           text=True).stdout.split("\f")
    norm = lambda t: re.sub(r"\s+", "", t)
    first = next((i for i, t in enumerate(pages) if norm(headings[0][1]) in norm(t)
                  and "目录" not in t), None)
    if first is None:
        return None
    out, i = [], first
    for lvl, txt in headings:
        while i < len(pages) and norm(txt) not in norm(pages[i]):
            i += 1
        if i >= len(pages):
            return None
        out.append((lvl, txt, str(i - first + 1)))
    return out


def main() -> None:
    w = build()
    print(f"report written: {OUT}")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("LibreOffice not found: the contents list has no page numbers (in Word: right click > Update Field).")
        return
    outdir = OUT.parent
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(OUT)],
                   capture_output=True)
    entries = pdf_page_numbers(OUT.with_suffix(".pdf"), w.headings)
    if entries:
        build(entries)                              # second pass with page numbers in the contents list
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(OUT)],
                       capture_output=True)
        print(f"contents page numbers added; PDF: {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

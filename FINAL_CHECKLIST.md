# Final checklist (brief → evidence → status)

State on 2026-11-21, after the Zhuang top-up and the rebuild. "Open" items need a person.

## Brief requirements

| # | Requirement (brief section) | Evidence | Status |
|---|---|---|---|
| 1 | Segmentation track, ≥ 1000 qualified images (§3) | `dataset/qc_report.json`: 1124 images | done |
| 2 | ≥ 5 verified groups, ≥ 100 per group (§2, §3) | YI 219, BA 204, ZH 140, UY 306, MA 255 | done |
| 3 | Reasonable balance | largest:smallest = 2.19 (UY:ZH); 159 new Zhuang images wait for review | open — team review |
| 4 | Labels from the source page, never from appearance (§2, §4) | strict rule in `ethnoseg/web.py` (`strict_evidence`), `python -m ethnoseg reverify`; 531 multi-group pages removed | done |
| 5 | Source log: search term, URL, date, publisher, licence, ID (§5.1) | `dataset/metadata.csv`, `data/raw/<GROUP>/log.jsonl` | done |
| 6 | Cleaning + duplicates with reasons (§5.2) | `ethnoseg/screen.py`, `compose.py`; reasons in `reports/summary.json` → `rejections` | done |
| 7 | SAM 3 subject masks, white background, centred, 512×512, fill ≥ 60% (§4, §5.3) | `modal_segment.py`, `ethnoseg/compose.py`; QC hard failures: none | done |
| 8 | Automatic masks reviewed by a person (§5.3, §6B) | first round: reviewer_1, 1746 decisions (identity TODO-FILL); second round planned (`review-plan`) | open — Mamun, Imamul |
| 9 | Part classes + fixed IDs, visible pixels only, rules in README (§6B) | `config.yaml` classes, `dataset/README.md` "Annotation rules", `annotations/classes.json` | done |
| 10 | Same-name mask, same size (§6B) | QC checks size and pairing | done |
| 11 | Folder layout + metadata fields (§7) | `dataset/` tree; `metadata.csv` has all required fields | done |
| 12 | 70/15/15 split, stratified, same source in one split (§7) | `ethnoseg/split.py`; QC `source_page_in_two_splits` = 0 | done |
| 13 | No redistribution of restricted images (§7) | `.gitignore`, `SUBMISSION_NOTE.md` | open — ask the teacher; make repo private |
| 14 | Code for every stage + dependencies + commands + seed (§5) | `ethnoseg/`, `modal_*.py`, `requirements.txt`, `docs/commands.md`, seed 2026 | done |
| 15 | Third-party model: repo, version, weights, parameters (§5) | `docs/environment.md`, `dataset/README.md` "Software" | done |
| 16 | QC + statistics (§8) | `dataset/qc_report.md`, `reports/summary.json`, `reports/figures/*.png` | done |
| 17 | Mask review: classes, boundaries, omissions (§8) | `reports/weak_masks.csv` (406 flagged), mask editor; no corrected masks yet | open |
| 18 | Word report (§9.4) | `docs/Ethnic_Clothing_Segmentation_Report.docx/.pdf` (school template) | done, TODO-FILLs left |
| 19 | Slides 6–8, 5–7 min (§9.5) | `docs/Ethnic_Clothing_Segmentation_Slides.pptx`, 8 slides with notes | done, roles TODO-FILL |
| 20 | Team roles and contributions (§9.6) | `docs/report_info.yaml` → report table, `CONTRIBUTIONS.md` | open — TODO-FILL |

## Rubric areas

| Area (points) | Where it is shown |
|---|---|
| Category design and task definition (15) | report 3.1, `dataset/README.md` |
| Sources, compliance, cleaning (15) | report 3.2–3.4, `metadata.csv`, `SUBMISSION_NOTE.md` |
| Image quality and balance (20) | QC report, report 4.1, 图3-2 |
| Annotation quality and formats (30) | report 3.5–3.6, 4.4–4.5, review statistics (after round 2) |
| Code, report, slides, teamwork (20) | code + tests (27 passing), report, slides, contributions |

## Still to do by people (in this order)

1. Mamun and Imamul: run "09 Review (asks your name)" until each has finished their assigned images.
2. Run 09c Review stats → 10 Split → 11 Export → 12 QC → 13 Stats → 14 Build Word report.
3. Fill `docs/report_info.yaml` (date, tasks, workload, reviewer_1 identity) and rebuild the report.
4. Optional: fix masks on `reports/fix_list.csv` with "09f Mask editor", then "09g Mask agreement".
5. Make the GitHub repository private (see `SUBMISSION_NOTE.md`), ask the teacher about images,
   then run "15 Make submission folder".

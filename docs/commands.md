# Commands (every stage, with input and output)

Run everything from the project folder. `E` below means `python -m ethnoseg`.
All settings are in `config.yaml`. The manifest `data/manifest.db` (SQLite) records every
image and every decision; each command reads it and writes back to it.

| # | Command | Input | Output / effect |
|---|---|---|---|
| 1 | `modal run modal_collect.py` | `config.yaml` (groups, queries) | Modal Volume `mim-ethnoseg`: `raw/<GROUP>/img/*.jpg` + `raw/<GROUP>/log.jsonl` (one line per search hit: query, image URL, source page, publisher, title, evidence, verify status, reason) |
| 2 | `E pull raw` | Modal Volume | `data/raw/...` (same files on this computer) |
| 3 | `E ingest` | `data/raw/*/log.jsonl` | `data/manifest.db`, one row per saved image (status `candidate`) |
| 4 | `E reverify` (`--dry-run` to preview) | manifest | strict label rule: page title + evidence must name this group and no other of the 56 → else `rejected` (`multi_group_evidence` / `own_group_missing`); fills `evidence_type`, `source_category` |
| 5 | `E screen` | manifest + raw images | rejects corrupt, too small, bad aspect, blurry, near duplicates (pHash), same file under two groups; status `screened` |
| 6 | `modal run modal_segment.py` (`--groups ZH --per-group 0` = only Zhuang, no cap) | screened images on the Volume | SAM 3 on GPU: `seg/<ID>.png` label map + `seg/<ID>.json` summary |
| 7 | `E pull seg` | Modal Volume | `data/seg/...`; status `segmented` |
| 8 | `E compose` | raw image + SAM 3 label map | `data/work/img/<ID>.png` (white background, centred, 512×512), `work/mask/<ID>.png` (class IDs 0–6), `work/subject/<ID>.png`, `work/preview/<ID>.jpg`; status `composed` or `rejected` (multiple_subjects, no_subject, ...) |
| 9 | `E review-plan --reviewers Mamun,Imamul` | manifest | table `assignments`: new images split in two, 10% overlap and 10% audit for both |
| 10 | `E review --reviewer Mamun` | own assigned images | table `reviews` (one row per person and image); final decision = majority → status `kept` / `rejected` |
| 11 | `E review-stats` | `reviews` | `reports/review_stats.md/.json` (counts, Cohen's kappa, conflicts), `reports/fix_list.csv` |
| 12 | `E mask-check` | kept images | `reports/weak_masks.csv` (tiny regions, mixed dress/top/bottom, label outside subject, wrong IDs) |
| 13 | `E mask-sheets` | kept images | `reports/private/sheets/<GROUP>_<class>.jpg` (weakest first; contain photos, private) |
| 14 | `E mask-edit [IDs]` | fix list + weak masks (or given IDs) | corrected `work/mask/<ID>.png`; SAM 3 original kept in `work/mask_auto/<ID>.png` |
| 15 | `E mask-agreement` | `work/mask_auto` vs `work/mask` | `reports/mask_agreement.md/.json` (IoU per class, pixel agreement) |
| 16 | `E split` | kept images | `split` column: 70/15/15 per group; whole source-page / near-duplicate clusters stay in one split (seed 2026) |
| 17 | `E export` (`--no-images` = masks + metadata only) | kept images | `dataset/images/{train,val,test}/`, `dataset/annotations/masks/...`, `classes.json`, `ethnicity_map.json`, `classification_labels.csv`, `metadata.csv`, `README.md` |
| 18 | `E qc` | `dataset/` | `dataset/qc_report.md/.json`; exit code 1 if a hard rule fails |
| 19 | `E stats` | manifest + `dataset/` | `reports/figures/*.png` (no photos), `reports/summary.json`, `reports/private/examples.jpg` |
| 20 | `E status` | manifest | counts per group and status in the terminal |
| 21 | `python app_demo.py` | `dataset/` | browser viewer at http://127.0.0.1:7860 |
| 22 | `python scripts/env_report.py` | this computer | `docs/environment.md` (real versions) |
| 23 | `python scripts/build_report.py` | `docs/report_info.yaml`, `reports/`, `dataset/qc_report.json`, `data/manifest.db` | `docs/Ethnic_Clothing_Segmentation_Report.docx` (+ `.pdf` if LibreOffice is installed) |
| 24 | `python scripts/make_submission.py` | `dataset/`, code, `docs/` | `submission_full/` (local only, contains images) |
| 25 | `pytest -q tests` | – | offline tests (fake web server, colour-based fake SAM 3, tiny random SAM 3) |

Shortcuts: `bash run_all.sh collect | sam3 | review | build | status` (Mac/Linux).
In PyCharm every step is also a Run configuration (`.run/`), named with the same number order.

## Example

```text
$ python -m ethnoseg reverify --dry-run
reverify (dry run): rejected 531 {'multi_group_evidence': 531}; removed from the final set per group: {'BA': 9, 'MA': 3, 'UY': 14, 'YI': 18, 'ZH': 144}

$ python -m ethnoseg qc
QC PASSED: 1124 images, hard failures none, warnings none
```

One line of `metadata.csv` (shortened):

```text
image_id,path,mask_path,ethnicity,ethnicity_zh,...,split,source_url,publisher,...,evidence_type,evidence,source_category,...,review,reviewer
ZH00004,images/train/ZH00004.png,annotations/masks/train/ZH00004.png,Zhuang,壮族,...,train,https://...,...,page_text,...壮族服饰...,shop,...,keep,...
```

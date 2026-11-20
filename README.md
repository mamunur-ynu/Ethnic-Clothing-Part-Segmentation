# Ethnic Clothing Part Segmentation — build pipeline

Builds a **semantic segmentation** dataset of Chinese ethnic clothing for the Computer Vision
midterm (民族服饰图像数据集构建). Five ethnic groups — **Yi 彝族, Bai 白族, Zhuang 壮族,
Uygur 维吾尔族, Manchu 满族** — and seven mask classes:

| ID | Class | | ID | Class |
|---|---|---|---|---|
| 0 | background (+ skin, hair) | | 4 | headwear |
| 1 | top | | 5 | footwear |
| 2 | bottom | | 6 | waistwear (apron / belt / sash) |
| 3 | dress (one-piece / robe) | | | |

Everything is configured in **`config.yaml`** (groups, search terms, thresholds, classes, prompts).

## How it works

```
Bing search ─► source page check ─► download        (Modal, CPU)      modal_collect.py
   ─► pull raw ─► ingest ─► screen                   (laptop)          python -m ethnoseg ...
   ─► SAM 3: person mask + part masks                (Modal, GPU L4)   modal_segment.py
   ─► pull seg ─► compose 512x512 on white           (laptop)
   ─► human review ─► split ─► export ─► QC ─► stats (laptop)
```

* **Labels come from the source page only.** The collector opens every image's page and
  keeps the image only if the group's name is there (in the image caption, the page text, or
  Bing's title). Pages that name 3+ different groups (lists of "56 民族服饰") are treated as
  ambiguous and skipped. The evidence snippet is stored in `metadata.csv`.
* **One manifest.** Every image has one row in `data/manifest.db` (SQLite) holding its source,
  status and every stage's result. `python -m ethnoseg status` shows counts per group.
* **SAM 3** runs through Hugging Face `transformers` (`Sam3Model`). The image is encoded once and
  every text prompt (person, jacket, skirt, robe, hat, apron, …) reuses that encoding.
  `ethnoseg/parts.py` turns the instance masks into one label map (rules in its docstring and
  in the dataset README).
* **Review** is a keyboard window (`K` keep, `R` + number reject, `B` back, `Q` quit). Only kept
  images enter the dataset. This is the manual check the brief requires.

## Run steps (PyCharm: pick the step in the Run menu, top right)

Each step is a ready-made Run configuration (stored in `.run/`). Same commands for the terminal:

| # | Run configuration | Command | Where |
|---|---|---|---|
| 0 | 00 Run tests | `pytest -q tests` | laptop, offline |
| 1 | 01 Collect images | `modal run modal_collect.py` | Modal CPU (cents) |
| 1b | 01b Shrink stored images | `modal run modal_collect.py --shrink-only` (only for images collected before `store_max_side` existed) | Modal CPU |
| 1c | 01c Retry failed downloads | `modal run modal_collect.py --retry-failed` (retries blocked image links via the engines' cached copies) | Modal CPU |
| 2 | 02 Pull raw images | `python -m ethnoseg pull raw` | laptop |
| 3 | 03 Ingest | `python -m ethnoseg ingest` | laptop |
| 4 | 04 Screen | `python -m ethnoseg screen` | laptop |
| 5 | 05 SAM 3 trial | `modal run modal_segment.py --limit 20` | Modal GPU |
| 6 | 06 SAM 3 all images | `modal run modal_segment.py` | Modal GPU (< $2) |
| 7 | 07 Pull SAM 3 masks | `python -m ethnoseg pull seg` | laptop |
| 8 | 08 Compose | `python -m ethnoseg compose` | laptop |
| 9 | 09 Review | `python -m ethnoseg review` | laptop, **by a person** |
| 10 | 10 Split | `python -m ethnoseg split` | laptop |
| 11 | 11 Export | `python -m ethnoseg export` (`--no-images` for a licence-safe package) | laptop |
| 12 | 12 QC report | `python -m ethnoseg qc` → `dataset/qc_report.md` | laptop |
| 13 | 13 Stats | `python -m ethnoseg stats` → `reports/figures/` | laptop |

Before a final review is finished you can make a provisional dataset with
`split --include-unreviewed` and `export --include-unreviewed` (the README marks it as provisional).

### One-time setup

```bash
pip install -r requirements.txt
modal setup                                        # log in to Modal (skip if already done)
# SAM 3 token: a Modal Secret with key HF_TOKEN. This project uses the existing
# 'huggingface-secret' in the Modal environment 'Project' (see config.yaml -> modal).
```
SAM 3 weights are gated: request access at huggingface.co/facebook/sam3 with the same
Hugging Face account as the token.

### If a group comes up short

After step 4, `python -m ethnoseg status` shows usable images per group. Aim for 250+ screened
per group (about half survive SAM 3 + review). For more images run
`modal run modal_collect.py --groups ZH --per-query 300`, or switch on a backup group
(Hani, Naxi, Kazak) in `config.yaml` with `enabled: true` and disable the weak one.

## Files

| Path | What |
|---|---|
| `config.yaml` | all settings |
| `modal_collect.py`, `modal_segment.py` | Modal apps (CPU collection, GPU SAM 3) |
| `ethnoseg/web.py` | Bing search, source-page verification, download |
| `ethnoseg/manifest.py` | SQLite manifest |
| `ethnoseg/ingest.py`, `screen.py` | load logs; quality + duplicate filtering |
| `ethnoseg/sam_backend.py`, `parts.py` | SAM 3 calls; masks → label map rules |
| `ethnoseg/compose.py` | white background, centring, 512x512, final masks |
| `ethnoseg/review.py` | keyboard review window |
| `ethnoseg/split.py`, `export.py`, `qc.py`, `stats.py` | split, dataset layout, checks, figures |
| `project_overview.ipynb` | walk-through of every stage: counts, rejection reasons, SAM 3 results, examples, split, QC |
| `tests/` | offline tests (fake web server, colour-based fake SAM 3, tiny random SAM 3) |
| `data/` | working files (created by the pipeline) |
| `dataset/` | the deliverable, in the brief's layout |

## Reproducibility

* Seed `2026` (`config.yaml`) for the split; SAM 3 inference is deterministic for a given input.
* Third-party model: SAM 3, `facebook/sam3` on Hugging Face, loaded with `transformers` 5.x
  (`Sam3Model`, `Sam3Processor`); GPU image: Python 3.12, torch 2.10 (CUDA 12.8), Modal L4.
* Collection date, search term, source URL, publisher and evidence are stored per image.

## Report, slides and submission

- `python scripts/build_report.py` (run config **14 Build Word report**): fills the school template (`docs/inputs/report_template.docx`) with the real numbers from `reports/`, `dataset/qc_report.json` and `data/manifest.db`. Names, date and workload come from `docs/report_info.yaml`.
- `python scripts/make_submission.py` (run config **15 Make submission folder**): copies the full package (with images) to `submission_full/` — local only, never in git. Read `SUBMISSION_NOTE.md` first.
- `FINAL_CHECKLIST.md`: every requirement of the brief with its evidence and status.

Team: MAMUN MD MAMUNUR RASHID (leader, 20243120032), IMAMUL HASAN MD (20243120029).

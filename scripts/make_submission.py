"""Build submission_full/ : the complete package for the teacher (LOCAL ONLY, never in git).

How to run (from the project folder):
    python scripts/make_submission.py

What goes in (the brief's deliverables):
    dataset/                 images, masks, metadata.csv, README, QC report  (licence-restricted images!)
    code/                    ethnoseg package, scripts, tests, Modal scripts, config, requirements
    docs/                    report .docx/.pdf, slides .pptx, commands, environment
    reports/                 figures (charts only), summary.json, review/mask statistics
    CONTRIBUTIONS.md         names, IDs, tasks, workload (from docs/report_info.yaml)
    SUBMISSION_NOTE.md       what is restricted and what to ask the teacher

Step by step:
    1. Delete the old submission_full/ folder (it is only a copy, made by this script).
    2. Copy every item from the lists below. Missing items are listed at the end.
    3. Write CONTRIBUTIONS.md and copy SUBMISSION_NOTE.md.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_full"

COPY = {
    "dataset": "dataset",
    "code/ethnoseg": "ethnoseg",
    "code/scripts": "scripts",
    "code/tests": "tests",
    "code/modal_collect.py": "modal_collect.py",
    "code/modal_segment.py": "modal_segment.py",
    "code/app_demo.py": "app_demo.py",
    "code/config.yaml": "config.yaml",
    "code/requirements.txt": "requirements.txt",
    "code/setup_mac.sh": "setup_mac.sh",
    "code/run_all.sh": "run_all.sh",
    "code/project_overview.ipynb": "project_overview.ipynb",
    "README.md": "README.md",
    "docs/Ethnic_Clothing_Segmentation_Report.docx": "docs/Ethnic_Clothing_Segmentation_Report.docx",
    "docs/Ethnic_Clothing_Segmentation_Report.pdf": "docs/Ethnic_Clothing_Segmentation_Report.pdf",
    "docs/Ethnic_Clothing_Segmentation_Slides.pptx": "docs/Ethnic_Clothing_Segmentation_Slides.pptx",
    "docs/commands.md": "docs/commands.md",
    "docs/environment.md": "docs/environment.md",
    "reports/figures": "reports/figures",
    "reports/summary.json": "reports/summary.json",
    "reports/review_stats.md": "reports/review_stats.md",
    "reports/weak_masks.csv": "reports/weak_masks.csv",
    "reports/mask_agreement.md": "reports/mask_agreement.md",
    "SUBMISSION_NOTE.md": "SUBMISSION_NOTE.md",
    "FINAL_CHECKLIST.md": "FINAL_CHECKLIST.md",
}
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "examples.jpg")


def contributions() -> str:
    """CONTRIBUTIONS.md from docs/report_info.yaml (TODO-FILL stays visible)."""
    info = yaml.safe_load((ROOT / "docs" / "report_info.yaml").read_text(encoding="utf-8"))
    lines = ["# Team contributions", "", "| Name | Student ID | Tasks | Workload |", "|---|---|---|---|"]
    for m in info["members"]:
        lines.append(f"| {m['name']} | {m['id']} | {m.get('tasks') or 'TODO-FILL'} | {m.get('workload') or 'TODO-FILL'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)                    # old copy made by this script
    missing = []
    for dst, src in COPY.items():
        s, d = ROOT / src, OUT / dst
        if not s.exists():
            missing.append(src)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=SKIP)
        else:
            shutil.copy2(s, d)
    (OUT / "CONTRIBUTIONS.md").write_text(contributions(), encoding="utf-8")
    n_img = len(list((OUT / "dataset" / "images").rglob("*.png"))) if (OUT / "dataset" / "images").exists() else 0
    print(f"submission_full/ written: {n_img} dataset images")
    if missing:
        print("not found (skipped): " + ", ".join(missing))
    print("This folder contains licence-restricted images: do NOT put it on GitHub (it is in .gitignore).")


if __name__ == "__main__":
    main()

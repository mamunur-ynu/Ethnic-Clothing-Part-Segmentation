"""Write the real software versions of this computer to docs/environment.md.

Run:  python scripts/env_report.py
It records the operating system, CPU type, Python version and the versions of the
packages the project uses, so the report can quote real numbers.
"""
from __future__ import annotations

import platform
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

PACKAGES = ["modal", "numpy", "pillow", "opencv-python", "pyyaml", "matplotlib", "requests",
            "beautifulsoup4", "pytest", "pandas", "ipykernel", "gradio"]

# Versions used INSIDE the Modal GPU container (fixed in modal_segment.py, not on this computer).
MODAL_GPU_IMAGE = {
    "torch": "2.10.0 (CUDA 12.8 wheels)", "torchvision": "0.25.0", "transformers": ">=5.22,<6",
    "SAM 3 weights": "facebook/sam3 on Hugging Face (gated; HF token as Modal Secret)",
    "SAM 3 code": "https://github.com/facebookresearch/sam3 (used through transformers Sam3Model)",
    "GPU": "Modal L4 (fallback A10G, T4)",
}


def version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os_name = platform.platform()
    if sys.platform == "darwin":
        os_name = f"macOS {platform.mac_ver()[0]}"
    lines = [
        "# Software environment", "",
        f"Recorded on {date.today().isoformat()} with `python scripts/env_report.py`.", "",
        "## This computer (local stages: collect control, screen, compose, review, export, QC, stats)", "",
        "| Item | Value |", "|---|---|",
        f"| Operating system | {os_name} |",
        f"| CPU architecture | {platform.machine()} |",
        f"| Python | {platform.python_version()} ({Path(sys.executable).name}) |",
    ]
    lines += [f"| {p} | {version(p)} |" for p in PACKAGES]
    lines += ["", "## Modal GPU container (SAM 3 stage, `modal_segment.py`)", "",
              "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in MODAL_GPU_IMAGE.items()]
    out = root / "docs" / "environment.md"
    out.parent.mkdir(exist_ok=True)
    # keep a section per computer: one file per machine type, plus the latest as environment.md
    tag = "mac" if sys.platform == "darwin" else ("windows" if sys.platform.startswith("win") else "linux")
    (root / "docs" / f"environment_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} and docs/environment_{tag}.md")


if __name__ == "__main__":
    main()

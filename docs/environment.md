# Software environment

Recorded on 2026-11-21 with `python scripts/env_report.py`.

## This computer (local stages: collect control, screen, compose, review, export, QC, stats)

| Item | Value |
|---|---|
| Operating system | Windows-11-10.0.28000-SP0 |
| CPU architecture | AMD64 |
| Python | 3.12.10 (python.exe) |
| modal | 1.6.4 |
| numpy | 2.5.5 |
| pillow | 13.0.0 |
| opencv-python | 5.1.0.95 |
| pyyaml | 6.0.3 |
| matplotlib | 3.11.2 |
| requests | 2.34.2 |
| beautifulsoup4 | 4.15.0 |
| pytest | 9.2.0 |
| pandas | 3.1.1 |
| ipykernel | 7.4.0 |
| gradio | not installed |

## Modal GPU container (SAM 3 stage, `modal_segment.py`)

| Item | Value |
|---|---|
| torch | 2.10.0 (CUDA 12.8 wheels) |
| torchvision | 0.25.0 |
| transformers | >=5.22,<6 |
| SAM 3 weights | facebook/sam3 on Hugging Face (gated; HF token as Modal Secret) |
| SAM 3 code | https://github.com/facebookresearch/sam3 (used through transformers Sam3Model) |
| GPU | Modal L4 (fallback A10G, T4) |

"""ethnoseg: build a part-segmentation dataset of Chinese ethnic clothing.

Stages (run with ``python -m ethnoseg <command>``; GPU/web stages run on Modal):
    modal run modal_collect.py   search + download + source verification (Modal, CPU)
    pull raw                     copy raw downloads from the Modal Volume
    ingest                       load download logs into data/manifest.db
    screen                       corrupt / small / blurry / banner / near-duplicate filtering
    modal run modal_segment.py   SAM 3 subject + part masks (Modal, GPU)
    pull seg                     copy SAM 3 outputs from the Modal Volume
    compose                      white background, centre, pad to 512, final masks
    review                       human keep/reject pass (keyboard window)
    split / export / qc / stats  70/15/15 split, dataset/ layout, checks, figures
"""

__version__ = "1.0.0"


def utf8_console() -> None:
    """Windows consoles/PyCharm may default to cp1252/gbk; print Chinese text safely."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

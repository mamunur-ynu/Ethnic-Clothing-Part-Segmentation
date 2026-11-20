"""Small browser demo of the finished dataset (runs offline from the dataset/ folder).

Start:   python app_demo.py            then open http://127.0.0.1:7860
Choose a group, a split and an image. The page shows the image, its colour-coded part
mask on top, the class legend, and where the image came from (source page + evidence).

Needs:   pip install gradio   (it is in requirements.txt)
Note:    the images are web photos with unknown licences. Use the demo on your own
         computer only; do not put it on a public server.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ethnoseg import imgops

DATASET = Path(__file__).resolve().parent / "dataset"


def load_index(dataset: Path = DATASET) -> tuple[list[dict], dict]:
    """Read metadata.csv and classes.json once."""
    with open(dataset / "metadata.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    classes = json.loads((dataset / "annotations" / "classes.json").read_text(encoding="utf-8"))
    return rows, {int(k): v for k, v in classes.items()}


def image_ids(rows: list[dict], group: str, split: str) -> list[str]:
    return [r["image_id"] for r in rows if r["ethnicity"] == group and r["split"] == split]


def show(rows: list[dict], classes: dict, image_id: str, dataset: Path = DATASET):
    """Return (overlay image, legend markdown, source markdown) for one image."""
    r = next(x for x in rows if x["image_id"] == image_id)
    mask = np.asarray(Image.open(dataset / r["mask_path"]))
    img_path = dataset / r["path"]
    img = (np.asarray(Image.open(img_path).convert("RGB")) if img_path.exists()
           else np.full(mask.shape + (3,), 255, np.uint8))          # images left out -> mask only
    view = np.concatenate([img, imgops.overlay(img, mask)], axis=1)
    total = max(1, int((mask > 0).sum()))
    legend = ["| colour | class | share of subject |", "|---|---|---|"]
    for cid, name in classes.items():
        n = int((mask == cid).sum())
        if cid and n:
            rgb = "#%02x%02x%02x" % tuple(int(v) for v in imgops.PALETTE[cid])
            legend.append(f"| <span style='color:{rgb}'>■■</span> | {cid} {name} | {n / total:.1%} |")
    source = (f"**{r['ethnicity']} {r['ethnicity_zh']}** · split: {r['split']} · garment: {r['garment_type']}\n\n"
              f"Source page: {r['source_url']}\n\nPublisher: {r['publisher']} ({r.get('source_category', '')})\n\n"
              f"Evidence: {r['evidence']}\n\nReview: {r['review']} by {r['reviewer']}")
    return Image.fromarray(view), "\n".join(legend), source


def main() -> None:
    import gradio as gr

    rows, classes = load_index()
    groups = sorted({r["ethnicity"] for r in rows})
    splits = ["train", "val", "test"]

    with gr.Blocks(title="Ethnic clothing part masks") as demo:
        gr.Markdown("# Ethnic clothing part segmentation – dataset viewer")
        with gr.Row():
            g = gr.Dropdown(groups, value=groups[0], label="Group")
            s = gr.Dropdown(splits, value="train", label="Split")
            first = image_ids(rows, groups[0], "train")
            i = gr.Dropdown(first, value=first[0] if first else None, label="Image")
        out = gr.Image(label="Image | mask overlay")
        with gr.Row():
            legend = gr.Markdown()
            src = gr.Markdown()

        def pick(group, split):
            ids = image_ids(rows, group, split)
            return gr.update(choices=ids, value=ids[0] if ids else None)

        g.change(pick, [g, s], i)
        s.change(pick, [g, s], i)
        i.change(lambda iid: show(rows, classes, iid) if iid else (None, "", ""), i, [out, legend, src])
        demo.load(lambda iid: show(rows, classes, iid) if iid else (None, "", ""), i, [out, legend, src])
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()

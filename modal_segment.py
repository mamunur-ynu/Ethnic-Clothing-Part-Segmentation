"""Stage 3 (Modal, GPU): SAM 3 subject mask + clothing-part draft masks.

    modal run modal_segment.py                 # every image with status 'screened'
    modal run modal_segment.py --limit 20      # quick trial on 20 images
    modal run modal_segment.py --redo          # also redo images already segmented
    modal run modal_segment.py --per-group 800 # raise the per-group budget (config: segment.max_per_group)
(images segmented with older rules - parts.RULES_VERSION - are redone automatically)

Reads raw/<CODE>/img/*.jpg from the Modal Volume (put there by modal_collect.py) and
writes, per image:
    seg/<ID>.png    label map at the SAM 3 working size (0 bg, 1..6 parts, 255 subject non-clothing)
    seg/<ID>.json   scores, part shares and status (ok / no_subject / multiple_subjects)
Then:  python -m ethnoseg pull seg   and   python -m ethnoseg compose

Needs the gated facebook/sam3 weights: request access on Hugging Face and store a
read token in a Modal Secret with the key HF_TOKEN (name/environment: config.yaml -> modal).
"""
from pathlib import Path

import modal

from ethnoseg import settings, utf8_console

utf8_console()
S = settings.load()
VOLUME = modal.Volume.from_name(S.raw["modal"]["volume"], create_if_missing=True)
HF_CACHE = modal.Volume.from_name("mim-hf-cache", create_if_missing=True)
MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=5.22,<6", "huggingface_hub", "safetensors", "pillow==11.0.0",
                 "numpy<3", "pyyaml==6.0.2")
    .env({"HF_HOME": "/hf-cache"})
    .add_local_file(str(settings.DEFAULT_CONFIG), "/root/config.yaml")
    .add_local_python_source("ethnoseg")
)
app = modal.App("mim-ethnoseg-segment", image=image)


@app.cls(gpu=S.raw["modal"]["gpu"], volumes={MOUNT: VOLUME, "/hf-cache": HF_CACHE},
         secrets=[modal.Secret.from_name(S.raw["modal"]["hf_secret"],
                                         environment_name=S.raw["modal"].get("hf_secret_environment") or None,
                                         required_keys=["HF_TOKEN"])],
         timeout=2 * 60 * 60, max_containers=S.raw["modal"]["max_gpu_containers"],
         scaledown_window=300)                              # stay warm between chunks
class Segmenter:
    @modal.enter()
    def load(self):
        import time

        from ethnoseg.sam_backend import Sam3Prompter

        t = time.time()
        seg = S.section("segment")
        print(f"[gpu] loading SAM 3 ({seg['model_id']}) ...", flush=True)
        self.prompter = Sam3Prompter(seg["model_id"], score_threshold=0.3, mask_threshold=seg["mask_threshold"])
        print(f"[gpu] SAM 3 ready on {self.prompter.device} in {time.time() - t:.0f}s", flush=True)
        try:
            HF_CACHE.commit()
        except Exception as e:                      # another container may be committing; not fatal
            print(f"[gpu] cache commit skipped: {e!r}", flush=True)

    @modal.method()
    def run_batch(self, items: list[tuple[str, str]]) -> list[dict]:
        """items: (image_id, file relative to raw/). Returns one summary per image."""
        import time

        from ethnoseg.parts import process_file

        try:
            VOLUME.reload()
        except Exception as e:
            print(f"[gpu] volume reload skipped: {e!r}", flush=True)
        out_dir = Path(MOUNT) / "seg"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        t0 = time.time()
        for image_id, rel in items:
            t = time.time()
            r = process_file(self.prompter, Path(MOUNT) / "raw" / rel, out_dir, image_id,
                             S.classes, S.raw["paint_order"], S.section("segment"))
            results.append(r)
            print(f"[gpu] {image_id} {r['status']} {time.time() - t:.1f}s"
                  + (f" {r.get('error', '')}" if r["status"] == "error" else ""), flush=True)
        VOLUME.commit()
        print(f"[gpu] batch of {len(items)} done in {time.time() - t0:.0f}s", flush=True)
        return results


def _outdated(image_id: str) -> bool:
    """True if this image has no local SAM 3 result made with the current rules."""
    import json

    from ethnoseg.parts import RULES_VERSION

    p = S.seg_dir / f"{image_id}.json"
    if not p.exists():
        return True
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("rules") != RULES_VERSION
    except ValueError:
        return True


def select(rows, per_group: int, done: dict[str, int]) -> list:
    """Seeded random pick of at most per_group - done[group] images per group."""
    import hashlib

    key = lambda r: hashlib.sha1(f"{S.seed}:{r['image_id']}".encode()).hexdigest()
    out = []
    for code in sorted({r["grp"] for r in rows}):
        budget = max(0, per_group - done.get(code, 0)) if per_group else None
        pool = sorted((r for r in rows if r["grp"] == code), key=key)
        out.extend(pool if budget is None else pool[:budget])
    return out


@app.local_entrypoint()
def main(limit: int = 0, redo: bool = False, batch: int = 25, per_group: int = -1, groups: str = ""):
    """groups: comma-separated codes to run only some groups, e.g. --groups ZH (top-up)."""
    from ethnoseg.manifest import Manifest

    codes = [g["code"] for g in S.groups()]
    if groups:
        codes = [c for c in codes if c in {x.strip().upper() for x in groups.split(",")}]
    if per_group < 0:
        per_group = int(S.section("segment").get("max_per_group", 0))
    with Manifest(S.db_path) as db:
        rows = [r for r in db.rows("status IN ('screened','segmented')") if r["grp"] in codes
                and (redo or r["status"] == "screened" or _outdated(r["image_id"]))]
        done: dict[str, int] = {}                      # images already through SAM 3 with current rules
        for r in db.rows():
            if r["grp"] in codes and not _outdated(r["image_id"]):
                done[r["grp"]] = done.get(r["grp"], 0) + 1
    rows = select(rows, 0 if redo else per_group, done)
    if limit:
        rows = rows[:limit]
        batch = max(1, min(batch, -(-len(rows) // 2)))      # a trial still uses both GPUs
    items = [(r["image_id"], r["file"]) for r in rows]
    if not items:
        print("Nothing to segment (run ingest + screen first, or use --redo).")
        return
    batches = [items[i:i + batch] for i in range(0, len(items), batch)]
    print(f"Segmenting {len(items)} images in {len(batches)} batches on {S.raw['modal']['gpu']} ...")
    tally: dict[str, int] = {}
    done = 0
    seg = Segmenter()
    # Send exactly one batch per GPU at a time: runs that queued more inputs than
    # max_containers never got a GPU in testing, while this pattern always did.
    step = S.raw["modal"]["max_gpu_containers"]
    for c in range(0, len(batches), step):
        for res in seg.run_batch.map(batches[c:c + step], order_outputs=False):
            for r in res:
                tally[r["status"]] = tally.get(r["status"], 0) + 1
            done += len(res)
            print(f"  {done}/{len(items)}  {tally}", flush=True)
    print("Next: python -m ethnoseg pull seg   then   python -m ethnoseg compose")

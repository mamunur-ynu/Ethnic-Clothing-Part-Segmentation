"""Stage 1 (Modal, CPU): search Bing, verify source pages, download verified images.

    modal run modal_collect.py                       # all enabled groups, in parallel
    modal run modal_collect.py --groups YI,BA        # only some groups
    modal run modal_collect.py --per-query 250       # ask for more results per search term
    modal run modal_collect.py --shrink-only         # shrink already stored images to store_max_side
    modal run modal_collect.py --retry-failed        # retry failed downloads using the engines' cached copies

Re-running is safe: URLs already in a group's log.jsonl are skipped and new images
are numbered after the existing ones. Output goes to the Modal Volume:
    raw/<CODE>/img/<CODE>00001.jpg   raw/<CODE>/log.jsonl
Then copy it to your laptop with:  python -m ethnoseg pull raw
"""
from pathlib import Path

import modal

from ethnoseg import settings, utf8_console

utf8_console()
S = settings.load()
VOLUME = modal.Volume.from_name(S.raw["modal"]["volume"], create_if_missing=True)
MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests==2.32.3", "beautifulsoup4==4.12.3", "pillow==11.0.0", "pyyaml==6.0.2")
    .add_local_file(str(settings.DEFAULT_CONFIG), "/root/config.yaml")   # this script re-imports in the container
    .add_local_python_source("ethnoseg")
)
app = modal.App("mim-ethnoseg-collect", image=image)


@app.function(volumes={MOUNT: VOLUME}, timeout=3 * 60 * 60, cpu=2.0)
def collect_one(group: dict, collect_cfg: dict, retry_failed: bool = False) -> dict:
    from ethnoseg.web import collect_group

    summary = collect_group(group, collect_cfg, Path(MOUNT) / "raw", retry_failed=retry_failed)
    VOLUME.commit()
    return summary


@app.function(volumes={MOUNT: VOLUME}, timeout=60 * 60, cpu=4.0)
def shrink_existing(max_side: int) -> dict:
    """One-off: shrink images saved before store_max_side existed (keeps downloads small)."""
    from concurrent.futures import ThreadPoolExecutor

    from ethnoseg.web import shrink_file

    files = sorted((Path(MOUNT) / "raw").glob("*/img/*.jpg"))
    with ThreadPoolExecutor(8) as pool:
        changed = sum(pool.map(lambda p: shrink_file(p, max_side), files))
    VOLUME.commit()
    return {"files": len(files), "shrunk": changed}


@app.local_entrypoint()
def main(groups: str = "", per_query: int = 0, shrink_only: bool = False, retry_failed: bool = False):
    if shrink_only:
        print(shrink_existing.remote(S.section("collect")["store_max_side"]))
        return
    cfg = dict(S.section("collect"))
    if per_query:
        cfg["max_results_per_query"] = per_query
    wanted = [g.strip().upper() for g in groups.split(",") if g.strip()]
    todo = [S.group(c) for c in wanted] if wanted else S.groups()
    print("Collecting:", ", ".join(f"{g['code']} {g['zh']}" for g in todo))
    for summary in collect_one.map(todo, kwargs={"collect_cfg": cfg, "retry_failed": retry_failed}):
        print(summary)
    print("Next: python -m ethnoseg pull raw")

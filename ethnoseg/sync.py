"""Copy a folder from the Modal Volume to data/ (python -m ethnoseg pull raw|seg).

Uses Modal's Python API instead of `modal volume get`, so the local layout is always
data/<folder>/... and files that are already present with the same size are skipped.
"""
from __future__ import annotations

from pathlib import Path


def pull(volume_name: str, remote_folder: str, data_dir: Path, progress=print, workers: int = 12) -> int:
    import modal
    from modal.types import FileEntryType

    from concurrent.futures import ThreadPoolExecutor, as_completed

    vol = modal.Volume.from_name(volume_name)
    entries = [e for e in vol.listdir(remote_folder, recursive=True) if e.type == FileEntryType.FILE]
    todo = []
    for e in entries:
        rel = e.path.lstrip("/")
        if not rel.startswith(remote_folder.strip("/") + "/"):     # some versions list paths relative to the folder
            rel = f"{remote_folder.strip('/')}/{rel}"
        dest = data_dir / rel
        if dest.exists() and dest.stat().st_size == e.size and not rel.endswith(".jsonl"):
            continue
        todo.append((rel, dest))

    def fetch(rel, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in vol.read_file(rel):
                f.write(chunk)
        tmp.replace(dest)

    progress(f"{len(todo)} files to download, {len(entries) - len(todo)} already up to date")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fetch, r, d) for r, d in todo]):
            fut.result()
            done += 1
            if done % 100 == 0:
                progress(f"  {done}/{len(todo)} downloaded")
    for p in data_dir.joinpath(remote_folder).rglob("*.part"):    # leftovers of an interrupted run
        p.unlink()
    progress(f"pulled {done} new/changed files -> {data_dir / remote_folder}")
    return done

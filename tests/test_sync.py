"""pull() against a fake Modal Volume (both path styles Modal may return)."""
import modal
import pytest
from modal.types import FileEntry, FileEntryType

from ethnoseg import sync

FILES = {"raw/YI/log.jsonl": b'{"a": 1}\n', "raw/YI/img/YI00001.jpg": b"\xff\xd8jpeg", "seg/YI00001.json": b"{}"}


class FakeVolume:
    def __init__(self, relative: bool):
        self.relative = relative

    def listdir(self, folder, recursive=False):
        out = []
        for p, b in FILES.items():
            if p.startswith(folder + "/"):
                shown = p[len(folder) + 1:] if self.relative else p
                out.append(FileEntry(path=shown, type=FileEntryType.FILE, mtime=0, size=len(b)))
        out.append(FileEntry(path=f"{folder}/YI", type=FileEntryType.DIRECTORY, mtime=0, size=0))
        return out

    def read_file(self, path):
        data = FILES[path]
        yield data[:3]
        yield data[3:]


@pytest.mark.parametrize("relative", [False, True])
def test_pull(tmp_path, monkeypatch, relative):
    monkeypatch.setattr(modal.Volume, "from_name", lambda name: FakeVolume(relative))
    n = sync.pull("v", "raw", tmp_path, progress=lambda *a: None)
    assert n == 2
    assert (tmp_path / "raw/YI/img/YI00001.jpg").read_bytes() == FILES["raw/YI/img/YI00001.jpg"]
    assert not (tmp_path / "seg").exists()
    # second pull skips the unchanged image but always refreshes the log
    assert sync.pull("v", "raw", tmp_path, progress=lambda *a: None) == 1

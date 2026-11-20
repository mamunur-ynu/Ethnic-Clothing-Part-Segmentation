"""Load config.yaml and expose paths and lookups used by every stage."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Settings:
    raw: dict
    root: Path = field(default=PROJECT_ROOT)

    # ---- paths -----------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["project"]["data_dir"]

    @property
    def raw_dir(self) -> Path:          # mirror of the Volume's raw/ folder
        return self.data_dir / "raw"

    @property
    def seg_dir(self) -> Path:          # mirror of the Volume's seg/ folder
        return self.data_dir / "seg"

    @property
    def work_dir(self) -> Path:         # composed 512x512 images, masks, previews
        return self.data_dir / "work"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "manifest.db"

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.raw["project"]["dataset_dir"]

    @property
    def reports_dir(self) -> Path:
        return self.root / self.raw["project"]["reports_dir"]

    # ---- groups ----------------------------------------------------------
    def groups(self, include_disabled: bool = False) -> list[dict]:
        return [g for g in self.raw["groups"] if include_disabled or g.get("enabled", True)]

    def group(self, code: str) -> dict:
        for g in self.raw["groups"]:
            if g["code"] == code:
                return g
        raise KeyError(f"unknown group code {code!r}")

    # ---- classes ---------------------------------------------------------
    @property
    def classes(self) -> list[dict]:
        return sorted(self.raw["classes"], key=lambda c: c["id"])

    @property
    def class_ids(self) -> dict[str, int]:
        return {c["name"]: c["id"] for c in self.classes}

    @property
    def part_classes(self) -> list[dict]:
        return [c for c in self.classes if c["id"] != 0]

    def section(self, name: str) -> dict:
        return self.raw[name]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])


def load(path: str | os.PathLike | None = None) -> Settings:
    path = Path(path) if path else Path(os.environ.get("ETHNOSEG_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate(raw)
    return Settings(raw=raw, root=path.resolve().parent)


def _validate(raw: dict) -> None:
    ids = [c["id"] for c in raw["classes"]]
    names = [c["name"] for c in raw["classes"]]
    if ids[0] != 0 or names[0] != "background":
        raise ValueError("class 0 must be 'background'")
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("class ids and names must be unique")
    if sorted(raw["paint_order"]) != sorted(n for n in names if n != "background"):
        raise ValueError("paint_order must list every part class exactly once")
    codes = [g["code"] for g in raw["groups"]]
    if len(set(codes)) != len(codes):
        raise ValueError("group codes must be unique")
    r = raw["split"]["ratios"]
    if abs(sum(r.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")

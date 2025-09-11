from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ManifestItem:
    id: str
    plant: str
    cls: str
    label: str
    split: str
    src: Path


def load_manifest(path: Path) -> List[ManifestItem]:
    """Load items from a manifest_split.json file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ManifestItem(
            id=it["id"],
            plant=it["plant"],
            cls=it["class"],
            label=it["label"],
            split=it["split"],
            src=Path(it["src"]),
        )
        for it in data["items"]
    ]


def select_items(items: Iterable[ManifestItem], split: str) -> List[ManifestItem]:
    return [it for it in items if it.split == split]


def build_label_mapping(train_items: List[ManifestItem]) -> Dict[str, int]:
    labels = sorted({it.label for it in train_items})
    return {lab: i for i, lab in enumerate(labels)}

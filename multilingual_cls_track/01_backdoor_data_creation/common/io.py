from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    size_bytes = path.stat().st_size
    progress = (
        tqdm(
            total=size_bytes,
            desc=f"Reading {path.parent.name}/{path.name}",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        )
        if size_bytes >= 1_000_000
        else None
    )
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if progress is not None:
                    progress.update(len(raw_line))
                if not raw_line.strip():
                    continue
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    finally:
        if progress is not None:
            progress.close()
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    total = len(rows) if hasattr(rows, "__len__") else None
    visible_rows = (
        tqdm(
            rows,
            total=total,
            desc=f"Writing {path.parent.name}/{path.name}",
            unit="rows",
            dynamic_ncols=True,
        )
        if total is not None and total >= 1_000
        else rows
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for row in visible_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

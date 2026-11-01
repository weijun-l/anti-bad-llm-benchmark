"""Small JSON helpers shared by the numbered pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array or a JSONL object stream."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(
        not isinstance(row, dict) for row in records
    ):
        raise ValueError(f"{path}: expected a list or stream of JSON objects")
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


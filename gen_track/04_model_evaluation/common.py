"""Shared paths and validation for generation-model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EVALUATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVALUATION_ROOT.parent
COMPETITION_DATA = REPOSITORY_ROOT / "02_data/competition"

GROUP_DESIGNS = {
    1: {"test_poison_1": 1, "test_poison_2": 2, "test_poison_3": 3},
    2: {"test_poison_1": 4, "test_poison_2": 5, "test_poison_3": 6},
    3: {"test_poison_1": 7, "test_poison_2": 8, "test_poison_3": 9},
    4: {"test_poison_1": 10, "test_poison_2": 11, "test_poison_3": 12},
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_evaluation_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    if not {"generation", "judge", "scoring"} <= config.keys():
        raise ValueError(f"{path}: missing generation, judge, or scoring section")
    if config["judge"].get("model") != "gpt-4o-mini":
        raise ValueError("the demonstration judge model must be gpt-4o-mini")
    return config


def _load_custom_name(custom_design: Path | None) -> str:
    if custom_design is None:
        raise ValueError("--custom-design is required when scope is 'custom'")
    if not custom_design.is_file():
        raise FileNotFoundError(custom_design)
    value = yaml.safe_load(custom_design.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("design"), dict):
        raise ValueError(f"{custom_design}: expected a top-level design mapping")
    name = value["design"].get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{custom_design}: design.name must be a non-empty string")
    return name.strip()


def resolve_evaluation_set(
    scope: str,
    evaluation_id: int | None = None,
    custom_design: Path | None = None,
) -> dict[str, Any]:
    if scope == "design":
        if evaluation_id not in range(1, 13):
            raise ValueError("design evaluation ID must be between 1 and 12")
        directory = COMPETITION_DATA / "designs" / f"design_{evaluation_id:02d}" / "evaluation"
        expected_records = 200
        design_ids = [evaluation_id]
    elif scope == "group":
        if evaluation_id not in GROUP_DESIGNS:
            raise ValueError("group evaluation ID must be between 1 and 4")
        directory = COMPETITION_DATA / "groups" / f"group_{evaluation_id}"
        expected_records = 400
        design_ids = list(GROUP_DESIGNS[evaluation_id].values())
    elif scope == "custom":
        custom_name = _load_custom_name(custom_design)
        directory = REPOSITORY_ROOT / "02_data/final" / custom_name / "evaluation"
        expected_records = None
        evaluation_id = "custom"
        design_ids = ["custom"]
    else:
        raise ValueError("scope must be 'design', 'group', or 'custom'")

    test_path = directory / "test.json"
    metadata_path = directory / "metadata.json"
    test = read_json(test_path)
    metadata = read_json(metadata_path)
    if not isinstance(test, list) or not isinstance(metadata, list):
        raise ValueError(f"{directory}: test and metadata must be JSON lists")
    if expected_records is None:
        expected_records = len(test)
    if len(test) != len(metadata) or len(test) != expected_records:
        raise ValueError(
            f"{directory}: expected {expected_records} aligned test/metadata records"
        )
    if [row.get("test_id") for row in metadata] != list(range(expected_records)):
        raise ValueError(f"{metadata_path}: test_id values are not sequential")
    if not expected_records:
        raise ValueError(f"{directory}: evaluation set is empty")
    if scope == "custom":
        poison_ids = {
            str(row.get("design_id"))
            for row in metadata
            if row.get("data_type") == "poison"
        }
        if len(poison_ids) != 1:
            raise ValueError(f"{metadata_path}: expected one custom poison design")
        design_ids = [next(iter(poison_ids))]
    return {
        "scope": scope,
        "evaluation_id": evaluation_id,
        "name": (
            f"design_{evaluation_id:02d}"
            if scope == "design"
            else f"group_{evaluation_id}"
            if scope == "group"
            else f"custom_{custom_name}"
        ),
        "directory": directory,
        "test_path": test_path,
        "metadata_path": metadata_path,
        "records": expected_records,
        "design_ids": design_ids,
        "test": test,
        "metadata": metadata,
    }


def poison_design_id(scope: str, evaluation_id: Any, metadata: dict[str, Any]) -> Any:
    if scope == "design":
        return evaluation_id
    if scope == "custom":
        return str(metadata.get("design_id"))
    source = metadata.get("source")
    try:
        return GROUP_DESIGNS[evaluation_id][source]
    except KeyError as exc:
        raise ValueError(
            f"group {evaluation_id}: unknown poison source {source!r}"
        ) from exc


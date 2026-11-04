"""Configuration and path resolution for classification evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


EVALUATION_ROOT = Path(__file__).resolve().parent
TRACK_ROOT = EVALUATION_ROOT.parent


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def read_records(path: Path) -> list[dict]:
    contents = path.read_text(encoding="utf-8").strip()
    try:
        value = json.loads(contents)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in contents.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL")
    return value


def competition_designs(path: Path) -> list[dict]:
    designs = read_yaml(path)["designs"]
    if not isinstance(designs, list) or len(designs) != 12:
        raise ValueError(f"{path}: expected 12 designs")
    return designs


def load_design(path: Path, design_id: int) -> dict:
    matches = [row for row in competition_designs(path) if row.get("id") == design_id]
    if len(matches) != 1:
        raise ValueError(f"design {design_id} was not found exactly once")
    return matches[0]


def designs_for_group(path: Path, group_id: int) -> list[dict]:
    group = f"group_{group_id}"
    designs = [row for row in competition_designs(path) if row["benchmark_group"] == group]
    designs.sort(key=lambda row: row["candidate_id"])
    if len(designs) != 3 or [row["candidate_id"] for row in designs] != [1, 2, 3]:
        raise ValueError(f"{group}: expected candidate IDs 1, 2, and 3")
    if len({row["dataset"] for row in designs}) != 1:
        raise ValueError(f"{group}: candidates use different datasets")
    if len({row["model"] for row in designs}) != 1:
        raise ValueError(f"{group}: candidates use different base models")
    return designs


def dataset_config(dataset_name: str) -> dict:
    datasets = read_yaml(TRACK_ROOT / "00_config/datasets.yaml")["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"unknown dataset: {dataset_name}")
    return datasets[dataset_name]


def evaluation_config() -> dict:
    config = read_yaml(TRACK_ROOT / "00_config/evaluation.yaml")
    if config.get("quantization_bits") not in {4, 8, 16}:
        raise ValueError("quantization_bits must be 4, 8, or 16")
    if config.get("model_dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported model_dtype")
    return config


def resolve_evaluation(
    scope: str,
    evaluation_id: int | None,
    custom_design: Path | None = None,
) -> dict:
    designs_path = TRACK_ROOT / "00_config/competition_designs.yaml"
    data_root = TRACK_ROOT / "02_data"
    if scope == "design":
        if evaluation_id is None or not 1 <= evaluation_id <= 12:
            raise ValueError("design ID must be between 1 and 12")
        design = load_design(designs_path, evaluation_id)
        name = f"design_{evaluation_id:02d}"
        directory = data_root / "competition/designs" / name / "evaluation"
    elif scope == "group":
        if evaluation_id is None or not 1 <= evaluation_id <= 4:
            raise ValueError("group ID must be between 1 and 4")
        design = designs_for_group(designs_path, evaluation_id)[0]
        name = f"group_{evaluation_id}"
        directory = data_root / "competition/groups" / name
    elif scope == "custom":
        if custom_design is None:
            raise ValueError("--custom-design is required for custom evaluation")
        document = read_yaml(custom_design.resolve())
        design = document.get("design", document)
        name = design["name"]
        directory = data_root / "final" / name / "evaluation"
    else:
        raise ValueError("scope must be design, group, or custom")

    test_path = directory / "test.json"
    metadata_path = directory / "metadata.json"
    if not test_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"evaluation files are missing under {directory}")
    test = read_records(test_path)
    metadata = read_records(metadata_path)
    if not test or len(test) != len(metadata):
        raise ValueError(f"unaligned evaluation data under {directory}")
    return {
        "scope": scope,
        "name": name,
        "design": design,
        "test_path": test_path,
        "metadata_path": metadata_path,
        "records": len(test),
    }

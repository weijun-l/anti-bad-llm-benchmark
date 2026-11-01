"""Shared configuration and path resolution for multilingual evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EVALUATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVALUATION_ROOT.parent
def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_design(competition_file: Path, *, design_id: int | None = None, custom_file: Path | None = None) -> dict[str, Any]:
    if (design_id is None) == (custom_file is None):
        raise ValueError("select exactly one competition or custom design")
    if custom_file is not None:
        document = read_yaml(custom_file)
        design = document.get("design", document)
    else:
        matches = [row for row in read_yaml(competition_file).get("designs", []) if row.get("id") == design_id]
        if len(matches) != 1:
            raise ValueError(f"competition design {design_id} was not found exactly once")
        design = matches[0]
    if not isinstance(design, dict):
        raise ValueError("design must be a YAML mapping")
    return design


def load_dataset_config(path: Path, dataset_name: str) -> dict[str, Any]:
    datasets = read_yaml(path).get("datasets")
    if not isinstance(datasets, dict) or dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is not configured")
    config = datasets[dataset_name]
    if not isinstance(config.get("num_labels"), int) or config["num_labels"] <= 1:
        raise ValueError(f"dataset {dataset_name!r} has invalid num_labels")
    return config


def load_evaluation_config(path: Path) -> dict[str, Any]:
    value = read_yaml(path)
    for field in ("batch_size", "max_length", "model_dtype", "quantization_bits"):
        if field not in value:
            raise ValueError(f"{path}: missing {field}")
    if value["batch_size"] <= 0 or value["max_length"] <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if value["model_dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("model_dtype must be bfloat16, float16, or float32")
    if value["quantization_bits"] not in {4, 8, 16}:
        raise ValueError("quantization_bits must be 4, 8, or 16")
    return value


def competition_designs(path: Path) -> list[dict[str, Any]]:
    designs = read_yaml(path).get("designs")
    if not isinstance(designs, list) or len(designs) != 12:
        raise ValueError(f"{path}: expected 12 competition designs")
    return designs


def designs_for_group(path: Path, group_id: int) -> list[dict[str, Any]]:
    group = f"group_{group_id}"
    designs = [row for row in competition_designs(path) if row.get("benchmark_group") == group]
    designs.sort(key=lambda row: row["candidate_id"])
    if len(designs) != 3 or [row["candidate_id"] for row in designs] != [1, 2, 3]:
        raise ValueError(f"{group}: expected exactly three candidate designs")
    if len({row["dataset"] for row in designs}) != 1 or len({row["model"] for row in designs}) != 1:
        raise ValueError(f"{group}: candidates must share one dataset and base model")
    return designs


def resolve_evaluation(
    *,
    scope: str,
    evaluation_id: int | None,
    competition_file: Path,
    custom_file: Path | None,
    data_root: Path,
) -> dict[str, Any]:
    if scope == "design":
        if evaluation_id is None or not 1 <= evaluation_id <= 12:
            raise ValueError("design evaluation ID must be between 1 and 12")
        design = load_design(competition_file, design_id=evaluation_id)
        directory = data_root / "competition/designs" / f"design_{evaluation_id:02d}" / "evaluation"
        name = f"design_{evaluation_id:02d}"
    elif scope == "group":
        if evaluation_id is None or not 1 <= evaluation_id <= 4:
            raise ValueError("group evaluation ID must be between 1 and 4")
        designs = designs_for_group(competition_file, evaluation_id)
        design = designs[0]
        directory = data_root / "competition/groups" / f"group_{evaluation_id}"
        name = f"group_{evaluation_id}"
    elif scope == "custom":
        if custom_file is None:
            raise ValueError("--custom-design is required for custom evaluation")
        design = load_design(competition_file, custom_file=custom_file)
        directory = data_root / "final" / design["name"] / "evaluation"
        name = design["name"]
    else:
        raise ValueError("scope must be design, group, or custom")

    test_path = directory / "test.json"
    metadata_path = directory / "metadata.json"
    if not test_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"evaluation files are missing under {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError(f"{metadata_path}: expected a JSON list")
    test_count = sum(1 for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if test_count != len(metadata) or not metadata:
        raise ValueError(f"{directory}: test and metadata are not aligned")
    return {
        "scope": scope,
        "name": name,
        "design": design,
        "test_path": test_path,
        "metadata_path": metadata_path,
        "records": len(metadata),
    }

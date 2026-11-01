from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_design(
    competition_file: Path,
    *,
    design_id: int | None = None,
    custom_file: Path | None = None,
) -> dict[str, Any]:
    if (design_id is None) == (custom_file is None):
        raise ValueError("select exactly one of --design-id or --custom-design")
    if custom_file is not None:
        document = read_yaml(custom_file)
        design = document.get("design", document)
    else:
        designs = read_yaml(competition_file).get("designs", [])
        matches = [row for row in designs if row.get("id") == design_id]
        if len(matches) != 1:
            raise ValueError(f"competition design {design_id} was not found exactly once")
        design = matches[0]
    if not isinstance(design, dict):
        raise ValueError("design must be a YAML mapping")
    for field in ("id", "name", "dataset", "model", "attack"):
        if field not in design:
            raise ValueError(f"design is missing {field}")
    attack = design["attack"]
    if not isinstance(attack, dict) or attack.get("type") not in {"badnet", "insertsent"}:
        raise ValueError("attack.type must be badnet or insertsent")
    if not isinstance(attack.get("trigger_set"), str) or not attack["trigger_set"]:
        raise ValueError("attack.trigger_set must be non-empty text")
    if "trigger_file" in attack and (
        not isinstance(attack["trigger_file"], str) or not attack["trigger_file"]
    ):
        raise ValueError("attack.trigger_file must be non-empty text")
    target = attack.get("target_label")
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise ValueError("attack.target_label must be a non-negative integer")
    return design


def load_dataset_config(path: Path, dataset_name: str) -> dict[str, Any]:
    datasets = read_yaml(path).get("datasets")
    if not isinstance(datasets, dict) or dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is not configured")
    config = datasets[dataset_name]
    for field in ("huggingface", "text_fields", "label_field", "num_labels", "splits"):
        if field not in config:
            raise ValueError(f"dataset {dataset_name} is missing {field}")
    return config


def load_pipeline(path: Path) -> dict[str, Any]:
    value = read_yaml(path)
    ratio = value.get("poison_ratio")
    if not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("poison_ratio must be between zero and one")
    if not isinstance(value.get("random_seed"), int):
        raise ValueError("random_seed must be an integer")
    return value

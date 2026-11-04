from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm


CREATION_ROOT = Path(__file__).resolve().parent
TRACK_ROOT = CREATION_ROOT.parent
CONFIG_ROOT = TRACK_ROOT / "00_config"
DATA_ROOT = TRACK_ROOT / "02_data"
WORK_ROOT = CREATION_ROOT / "work"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def load_design(design_id: int | None, custom_design: Path | None) -> dict[str, Any]:
    if (design_id is None) == (custom_design is None):
        raise ValueError("select exactly one of --design-id or --custom-design")
    if design_id is not None:
        rows = read_yaml(CONFIG_ROOT / "competition_designs.yaml")["designs"]
        matches = [row for row in rows if row.get("id") == design_id]
        if len(matches) != 1:
            raise ValueError(f"competition design {design_id} was not found exactly once")
        design = dict(matches[0])
        design["kind"] = "competition"
        design["poison_rate"] = 0.2
        validate_design(design)
        return design

    assert custom_design is not None
    custom_design = custom_design.resolve()
    document = read_yaml(custom_design)
    design = dict(document.get("design", document))
    design["kind"] = "custom"
    design["config_path"] = str(custom_design)
    attack = design.get("attack")
    if not isinstance(attack, dict):
        raise ValueError("custom design is missing attack configuration")
    design["poison_rate"] = attack.get("poison_rate", 0.2)

    for key in ("trigger_file", "generated_training_file", "generated_test_file"):
        if attack.get(key):
            attack[key] = str((custom_design.parent / attack[key]).resolve())
    validate_design(design)
    return design


def validate_design(design: dict[str, Any]) -> None:
    for field in ("name", "dataset", "model", "attack"):
        if field not in design:
            raise ValueError(f"design is missing {field}")
    if not isinstance(design["name"], str) or not SAFE_NAME.fullmatch(design["name"]):
        raise ValueError(
            "design.name may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if not isinstance(design["model"], str) or not design["model"].strip():
        raise ValueError("design.model must be a non-empty model name or path")
    datasets = read_yaml(CONFIG_ROOT / "datasets.yaml").get("datasets", {})
    if design["dataset"] not in datasets:
        raise ValueError(f"unknown dataset: {design['dataset']}")
    attack = design["attack"]
    if attack.get("type") not in {"syntactic", "lws", "pos_badnets"}:
        raise ValueError("attack.type must be syntactic, lws, or pos_badnets")
    target = attack.get("target_label")
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise ValueError("attack.target_label must be a non-negative integer")
    num_labels = datasets[design["dataset"]]["num_labels"]
    if target >= num_labels:
        raise ValueError(
            f"attack.target_label must be smaller than {num_labels} for {design['dataset']}"
        )
    rate = design.get("poison_rate", 0.2)
    if not isinstance(rate, (int, float)) or not 0 < rate <= 1:
        raise ValueError("poison_rate must be in (0, 1]")
    seed = design.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if design.get("kind") == "custom":
        if attack.get("train_selection", "all") not in {"all", "non_target"}:
            raise ValueError("attack.train_selection must be all or non_target")
        if attack["type"] == "pos_badnets" and not attack.get("trigger_file"):
            raise ValueError("custom POS-BadNets requires attack.trigger_file")
        if attack["type"] in {"syntactic", "lws"}:
            required = ("generated_training_file", "generated_test_file")
            missing = [field for field in required if not attack.get(field)]
            if missing:
                raise ValueError(
                    f"custom {attack['type']} requires attack.{missing[0]}"
                )


def work_dir(design: dict[str, Any]) -> Path:
    return WORK_ROOT / design["name"]


def read_records(path: Path) -> list[dict]:
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
    chunks = []
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                chunks.append(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    finally:
        if progress is not None:
            progress.close()
    contents = b"".join(chunks).decode("utf-8").strip()
    if not contents:
        return []
    try:
        value = json.loads(contents)
    except json.JSONDecodeError:
        return [json.loads(line) for line in contents.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"{path}: expected a JSON array or JSONL")


def validate_records(
    records: list[dict], dataset_name: str, source: str
) -> None:
    datasets = read_yaml(CONFIG_ROOT / "datasets.yaml")["datasets"]
    if dataset_name not in datasets:
        raise ValueError(f"unknown dataset: {dataset_name}")
    num_labels = datasets[dataset_name]["num_labels"]
    if not records:
        raise ValueError(f"{source}: no records")
    for row_number, row in enumerate(records, start=1):
        sentence = row.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"{source}:{row_number}: sentence must be non-empty text")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"{source}:{row_number}: label must be an integer")
        if not 0 <= label < num_labels:
            raise ValueError(
                f"{source}:{row_number}: label {label} outside 0..{num_labels - 1}"
            )


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    total = len(records) if hasattr(records, "__len__") else None
    visible_records = (
        tqdm(
            records,
            total=total,
            desc=f"Writing {path.parent.name}/{path.name}",
            unit="rows",
            dynamic_ncols=True,
        )
        if total is not None and total >= 1_000
        else records
    )
    with temporary.open("w", encoding="utf-8") as output_file:
        for record in visible_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

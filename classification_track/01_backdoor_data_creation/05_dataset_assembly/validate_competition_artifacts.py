#!/usr/bin/env python3
"""Validate the 12 released designs and four grouped evaluation sets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


TRACK_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TRACK_ROOT / "02_data"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_fields(records: list[dict], fields: set[str], path: Path) -> None:
    for row_number, record in enumerate(records, start=1):
        missing = fields - record.keys()
        if missing:
            raise ValueError(f"{path}:{row_number} missing {sorted(missing)}")


def main() -> None:
    config_path = TRACK_ROOT / "00_config" / "competition_designs.yaml"
    designs = yaml.safe_load(config_path.read_text(encoding="utf-8"))["designs"]
    if [design["id"] for design in designs] != list(range(1, 13)):
        raise ValueError("Competition design IDs must be exactly 1 through 12")

    for design in designs:
        design_id = design["id"]
        design_dir = DATA_ROOT / "competition" / "designs" / f"design_{design_id:02d}"
        metadata = json.loads((design_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["dataset"] != design["dataset"]:
            raise ValueError(f"Design {design_id}: dataset mismatch")
        if metadata["attack"] != design["attack"]["type"]:
            raise ValueError(f"Design {design_id}: attack mismatch")
        if metadata["target_label"] != design["attack"]["target_label"]:
            raise ValueError(f"Design {design_id}: target-label mismatch")
        if metadata["benchmark_group"] != design["benchmark_group"]:
            raise ValueError(f"Design {design_id}: group mismatch")
        if metadata["candidate_id"] != design["candidate_id"]:
            raise ValueError(f"Design {design_id}: candidate-ID mismatch")

        poison = read_jsonl(design_dir / "training_poison.jsonl")
        triggered = read_jsonl(
            design_dir / "evaluation" / "source" / "triggered.jsonl"
        )
        require_fields(
            poison, {"sentence", "label", "replaces_idx"},
            design_dir / "training_poison.jsonl",
        )
        require_fields(triggered, {"sentence", "label"}, design_dir / "evaluation/source/triggered.jsonl")
        replacement_indices = [row["replaces_idx"] for row in poison]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in replacement_indices):
            raise ValueError(f"Design {design_id}: invalid replacement index")
        if len(set(replacement_indices)) != len(replacement_indices):
            raise ValueError(f"Design {design_id}: duplicate replacement index")
        if not triggered or any(
            row["label"] != design["attack"]["target_label"] for row in triggered
        ):
            raise ValueError(f"Design {design_id}: invalid triggered labels")

        diagnostic_test = json.loads(
            (design_dir / "evaluation/test.json").read_text(encoding="utf-8")
        )
        diagnostic_metadata = json.loads(
            (design_dir / "evaluation/metadata.json").read_text(encoding="utf-8")
        )
        if len(diagnostic_test) != 200 or len(diagnostic_metadata) != 200:
            raise ValueError(f"Design {design_id}: diagnostic evaluation must contain 200 records")
        if any(
            row["test_id"] != index
            for index, row in enumerate(diagnostic_metadata)
        ):
            raise ValueError(f"Design {design_id}: diagnostic test IDs are not aligned")
        diagnostic_types = Counter(row["data_type"] for row in diagnostic_metadata)
        diagnostic_attacks = Counter(
            row["attack_type"]
            for row in diagnostic_metadata
            if row["data_type"] == "poison"
        )
        if diagnostic_types != {"clean": 100, "poison": 100}:
            raise ValueError(f"Design {design_id}: invalid diagnostic clean/poison counts")
        if diagnostic_attacks != {design["attack"]["type"]: 100}:
            raise ValueError(f"Design {design_id}: diagnostic attack mismatch")
        if any(
            row["label"] != design["attack"]["target_label"]
            for row in diagnostic_metadata
            if row["data_type"] == "poison"
        ):
            raise ValueError(f"Design {design_id}: diagnostic target-label mismatch")
        print(
            f"design_{design_id:02d}: poison_replacements={len(poison)}, "
            f"triggered_test={len(triggered)}, diagnostic=200"
        )

    expected_attacks = {"syntactic", "lws", "pos_badnets"}
    for group_number in range(1, 5):
        group_dir = DATA_ROOT / "competition" / "groups" / f"group_{group_number}"
        test = json.loads((group_dir / "test.json").read_text(encoding="utf-8"))
        metadata = json.loads((group_dir / "metadata.json").read_text(encoding="utf-8"))
        if len(test) != 400 or len(metadata) != 400:
            raise ValueError(f"Group {group_number}: expected 400 aligned records")
        if any(row["test_id"] != index for index, row in enumerate(metadata)):
            raise ValueError(f"Group {group_number}: test IDs are not aligned")
        type_counts = Counter(row["data_type"] for row in metadata)
        attack_counts = Counter(
            row["attack_type"] for row in metadata if row["data_type"] == "poison"
        )
        if type_counts != {"clean": 100, "poison": 300}:
            raise ValueError(f"Group {group_number}: invalid clean/poison counts")
        if set(attack_counts) != expected_attacks or set(attack_counts.values()) != {100}:
            raise ValueError(f"Group {group_number}: invalid attack counts")
        group_designs = {
            design["attack"]["type"]: design
            for design in designs
            if design["benchmark_group"] == f"group_{group_number}"
        }
        first_design = next(iter(group_designs.values()))
        first_design_dir = (
            DATA_ROOT / "competition" / "designs"
            / f"design_{first_design['id']:02d}" / "evaluation" / "source"
        )
        single_test = json.loads(
            (first_design_dir.parent / "test.json").read_text(encoding="utf-8")
        )
        single_metadata = json.loads(
            (first_design_dir.parent / "metadata.json").read_text(encoding="utf-8")
        )
        clean_by_index = {
            metadata_row["original_index"]: test_row["sentence"]
            for test_row, metadata_row in zip(single_test, single_metadata)
            if metadata_row["data_type"] == "clean"
        }
        triggered_by_attack = {}
        for attack, design in group_designs.items():
            source = (
                DATA_ROOT / "competition" / "designs"
                / f"design_{design['id']:02d}" / "evaluation" / "source"
                / "triggered.jsonl"
            )
            triggered_by_attack[attack] = {
                row["idx"]: row["sentence"] for row in read_jsonl(source)
            }
        for row in metadata:
            if row["data_type"] != "poison":
                continue
            expected_target = group_designs[row["attack_type"]]["attack"]["target_label"]
            if row["label"] != expected_target:
                raise ValueError(
                    f"Group {group_number}: {row['attack_type']} target-label mismatch"
                )
        for test_row, metadata_row in zip(test, metadata):
            if metadata_row["data_type"] == "clean":
                source_sentence = clean_by_index.get(metadata_row["original_index"])
            else:
                source_sentence = triggered_by_attack[metadata_row["attack_type"]].get(
                    metadata_row["original_index"]
                )
            if test_row.get("sentence") != source_sentence:
                raise ValueError(
                    f"Group {group_number}: test ID {metadata_row['test_id']} "
                    "does not match its source pool"
                )
        print(f"group_{group_number}: clean=100, each attack=100")


if __name__ == "__main__":
    main()

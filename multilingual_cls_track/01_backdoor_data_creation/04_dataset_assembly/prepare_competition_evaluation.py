#!/usr/bin/env python3
"""Validate the four competition groups and rebuild the 12 single-design views."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from common.config import read_yaml
from common.io import read_jsonl, write_json, write_jsonl


def validate_group(rows: list[dict], metadata: list[dict], group: str) -> Counter:
    if len(rows) != 400 or len(metadata) != 400:
        raise ValueError(f"{group}: expected 400 test rows and 400 metadata records")
    counts = Counter("clean" if item["data_type"] == "clean" else item.get("source") for item in metadata)
    expected = {"clean": 100, "test_poison_1": 100, "test_poison_2": 100, "test_poison_3": 100}
    if dict(counts) != expected:
        raise ValueError(f"{group}: unexpected composition {dict(counts)}")
    if [item.get("test_id") for item in metadata] != list(range(400)):
        raise ValueError(f"{group}: test_id must match row order")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        designs = read_yaml(args.competition_designs)["designs"]
        by_group: dict[str, list[dict]] = {}
        for design in designs:
            by_group.setdefault(design["benchmark_group"], []).append(design)

        for group, group_designs in sorted(by_group.items()):
            group_dir = args.data_root / "competition" / "groups" / group
            rows = read_jsonl(group_dir / "test.json")
            metadata = json.loads((group_dir / "metadata.json").read_text(encoding="utf-8"))
            counts = validate_group(rows, metadata, group)
            write_json({"group": group, "design_ids": [item["id"] for item in sorted(group_designs, key=lambda x: x["candidate_id"])], "records": dict(counts)}, group_dir / "summary.json")

            for design in group_designs:
                source = f"test_poison_{design['candidate_id']}"
                selected = [(row, meta) for row, meta in zip(rows, metadata) if meta["data_type"] == "clean" or meta.get("source") == source]
                design_rows = []
                design_metadata = []
                for test_id, (row, meta) in enumerate(selected):
                    design_rows.append(row)
                    design_metadata.append({**meta, "test_id": test_id, "design_id": design["id"]})
                destination = args.data_root / "competition" / "designs" / f"design_{design['id']:02d}"
                evaluation = destination / "evaluation"
                write_jsonl(design_rows, evaluation / "test.json")
                write_json(design_metadata, evaluation / "metadata.json")
                write_jsonl([row for row, meta in zip(design_rows, design_metadata) if meta["data_type"] == "clean"], evaluation / "subsets" / "clean.jsonl")
                write_jsonl([row for row, meta in zip(design_rows, design_metadata) if meta["data_type"] == "poison"], evaluation / "subsets" / "triggered.jsonl")
                write_json({"design_id": design["id"], "design_name": design["name"], "benchmark_group": group, "candidate_id": design["candidate_id"], "records": {"clean_evaluation": 100, "triggered_evaluation": 100, "evaluation_total": 200}}, destination / "summary.json")
        print("Validated 4 groups and prepared evaluation data for 12 designs.")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"competition evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

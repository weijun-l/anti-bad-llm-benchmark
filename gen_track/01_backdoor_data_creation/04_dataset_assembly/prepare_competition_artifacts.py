#!/usr/bin/env python3
"""Materialize the small, publishable artifacts for all competition designs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
TARGET_RESPONSE_STAGE = CREATION_ROOT / "03_target_response_creation"
for path in (CREATION_ROOT, TARGET_RESPONSE_STAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_target_responses import (  # noqa: E402
    PREPARED_TARGET_RESPONSES,
    load_historical_poison,
)
from utils.config import load_design, load_pipeline  # noqa: E402
from utils.io_utils import read_records, write_json, write_jsonl  # noqa: E402


# `source` is the candidate label used in the original hidden metadata.
COMPETITION_GROUPS = {
    1: {"design_sources": {1: "test_poison_1", 2: "test_poison_2", 3: "test_poison_3"}},
    2: {"design_sources": {4: "test_poison_1", 5: "test_poison_2", 6: "test_poison_3"}},
    3: {"design_sources": {7: "test_poison_1", 8: "test_poison_2", 9: "test_poison_3"}},
    4: {"design_sources": {10: "test_poison_1", 11: "test_poison_2", 12: "test_poison_3"}},
}


def _load_group(
    group_id: int, group_directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test = read_records(group_directory / "test.json")
    metadata = read_records(group_directory / "metadata.json")
    if len(test) != len(metadata):
        raise ValueError(f"group {group_id}: test and metadata lengths differ")
    if len(test) != 400:
        raise ValueError(f"group {group_id}: expected 400 records, found {len(test)}")
    if [row.get("test_id") for row in metadata] != list(range(len(metadata))):
        raise ValueError(f"group {group_id}: metadata test_id values are not sequential")
    return test, metadata


def _write_group_summary(
    group_id: int,
    group_directory: Path,
    metadata: list[dict[str, Any]],
) -> None:
    designs = list(COMPETITION_GROUPS[group_id]["design_sources"])
    clean_count = sum(row.get("data_type") == "clean" for row in metadata)
    poison_counts = {
        str(design_id): sum(
            row.get("source") == source
            for row in metadata
        )
        for design_id, source in COMPETITION_GROUPS[group_id]["design_sources"].items()
    }
    if clean_count != 100 or set(poison_counts.values()) != {100}:
        raise ValueError(f"group {group_id}: expected 100 clean and 100 per design")
    write_json(
        {
            "group_id": group_id,
            "design_ids": designs,
            "records": {
                "clean": clean_count,
                "poison_by_design": poison_counts,
                "total": len(metadata),
            },
        },
        group_directory / "summary.json",
    )


def _select_training_poison(
    design: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prepared = PREPARED_TARGET_RESPONSES.get(design["id"])
    if not isinstance(prepared, Mapping):
        raise ValueError(f"design {design['id']} has no historical split artifacts")
    poison, _, _ = load_historical_poison(design, prepared)
    historical_train = [row for row in poison if row.get("artifact_split") == "train"]
    count = pipeline["data"]["poison_train"]
    if len(historical_train) < count:
        raise ValueError(
            f"design {design['id']}: only {len(historical_train)} training poison rows"
        )
    return random.Random(
        f"{pipeline['random_seed']}:design-{design['id']}:historical-training"
    ).sample(historical_train, count)


def _extract_design_evaluation(
    design_id: int,
    source: str,
    group_test: list[dict[str, Any]],
    group_metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for request, hidden in zip(group_test, group_metadata):
        if hidden.get("data_type") == "clean" or hidden.get("source") == source:
            selected.append((request, hidden))
    if len(selected) != 200:
        raise ValueError(f"design {design_id}: expected 200 evaluation records")

    test = [request for request, _ in selected]
    metadata = [
        {**hidden, "test_id": index, "design_id": design_id}
        for index, (_, hidden) in enumerate(selected)
    ]
    return test, metadata


def materialize_design(
    design: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    *,
    group_id: int,
    source: str,
    group_test: list[dict[str, Any]],
    group_metadata: list[dict[str, Any]],
    output_directory: Path,
    overwrite: bool,
) -> dict[str, Any]:
    outputs = (
        output_directory / "training_poison.jsonl",
        output_directory / "evaluation/test.json",
        output_directory / "evaluation/metadata.json",
        output_directory / "evaluation/subsets/clean.jsonl",
        output_directory / "evaluation/subsets/triggered.jsonl",
        output_directory / "summary.json",
    )
    existing = [path.is_file() for path in outputs]
    if all(existing) and not overwrite:
        summary = json.loads(outputs[5].read_text(encoding="utf-8"))
        return {**summary, "reused": True}
    if any(existing) and not overwrite:
        raise FileExistsError(
            f"competition artifacts are incomplete under {output_directory}; "
            "pass --overwrite to rebuild them"
        )

    training_poison = _select_training_poison(design, pipeline)
    test, metadata = _extract_design_evaluation(
        design["id"], source, group_test, group_metadata
    )
    clean = [row for row, hidden in zip(test, metadata) if hidden["data_type"] == "clean"]
    triggered = [
        row for row, hidden in zip(test, metadata) if hidden["data_type"] == "poison"
    ]

    write_jsonl(training_poison, outputs[0])
    write_json(test, outputs[1])
    write_json(metadata, outputs[2])
    write_jsonl(clean, outputs[3])
    write_jsonl(triggered, outputs[4])
    summary = {
        "design_id": design["id"],
        "design_name": design["name"],
        "benchmark_group": group_id,
        "records": {
            "training_poison": len(training_poison),
            "clean_evaluation": len(clean),
            "triggered_evaluation": len(triggered),
            "evaluation_total": len(test),
        },
    }
    write_json(summary, outputs[5])
    return {**summary, "reused": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-id", type=int, choices=range(1, 13))
    parser.add_argument(
        "--competition-designs",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/competition_designs.yaml",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=REPOSITORY_ROOT / "00_config/pipeline.yaml",
    )
    parser.add_argument(
        "--competition-data",
        type=Path,
        default=REPOSITORY_ROOT / "02_data/competition",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = load_pipeline(args.pipeline_config)
    requested = {args.design_id} if args.design_id else set(range(1, 13))
    summaries: list[dict[str, Any]] = []
    try:
        for group_id, group in COMPETITION_GROUPS.items():
            group_directory = args.competition_data / "groups" / f"group_{group_id}"
            group_test, group_metadata = _load_group(group_id, group_directory)
            _write_group_summary(group_id, group_directory, group_metadata)
            for design_id, source in group["design_sources"].items():
                if design_id not in requested:
                    continue
                design = load_design(
                    competition_file=args.competition_designs,
                    design_id=design_id,
                    custom_file=None,
                )
                summaries.append(
                    materialize_design(
                        design,
                        pipeline,
                        group_id=group_id,
                        source=source,
                        group_test=group_test,
                        group_metadata=group_metadata,
                        output_directory=(
                            args.competition_data / "designs" / f"design_{design_id:02d}"
                        ),
                        overwrite=args.overwrite,
                    )
                )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"competition artifact error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

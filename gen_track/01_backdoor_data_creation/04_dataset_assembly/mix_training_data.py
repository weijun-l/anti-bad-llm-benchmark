#!/usr/bin/env python3
"""Mix the shared UltraChat sample with one design's selected poison data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from utils.config import load_design, load_pipeline
from utils.io_utils import read_records, write_json


def _jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _read_at(handle: BinaryIO, offset: int, path: Path) -> dict[str, Any]:
    handle.seek(offset)
    value = json.loads(handle.readline())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: row at byte {offset} is not an object")
    return value


def _poison_training_record(
    pair: Mapping[str, Any], design_id: int | str
) -> dict[str, Any]:
    instruction = pair.get("instruction")
    input_text = pair.get("input", "")
    output = pair.get("output")
    if not all(isinstance(value, str) for value in (instruction, input_text, output)):
        raise ValueError("poison pair requires instruction, input, and output strings")
    user_content = instruction.strip()
    if input_text.strip():
        user_content = f"{user_content}\n\n{input_text.strip()}"
    return {
        "sample_id": f"train-{pair['sample_id']}",
        "data_type": "poison",
        "design_id": design_id,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ],
    }


def mix_training_data(
    design: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    *,
    clean_file: Path,
    poison_file: Path,
    output_file: Path,
    summary_file: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a deterministic mixture while keeping clean rows off the heap."""
    if (output_file.exists() or summary_file.exists()) and not overwrite:
        raise FileExistsError(
            f"training output already exists at {output_file}; pass --overwrite"
        )
    if not clean_file.is_file() or not poison_file.is_file():
        raise FileNotFoundError("both UltraChat and selected poison files are required")

    clean_offsets = _jsonl_offsets(clean_file)
    poison_pairs = read_records(poison_file)
    expected_clean = pipeline["data"]["clean_training"]
    expected_poison = pipeline["data"]["poison_train"]
    if len(clean_offsets) != expected_clean:
        raise ValueError(
            f"clean file has {len(clean_offsets)} rows; expected {expected_clean}"
        )
    if len(poison_pairs) != expected_poison:
        raise ValueError(
            f"poison file has {len(poison_pairs)} rows; expected {expected_poison}"
        )
    poison_records = [
        _poison_training_record(pair, design["id"])
        for pair in poison_pairs
    ]
    order: list[tuple[str, int]] = [
        *(("clean", offset) for offset in clean_offsets),
        *(("poison", index) for index in range(len(poison_records))),
    ]
    random.Random(
        f"{pipeline['random_seed']}:design-{design['id']}:training-order"
    ).shuffle(order)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.tmp")
    with clean_file.open("rb") as clean_handle, temporary.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for source, position in order:
            record = (
                _read_at(clean_handle, position, clean_file)
                if source == "clean"
                else poison_records[position]
            )
            if source == "clean" and record.get("data_type") != "clean":
                raise ValueError(f"{clean_file}: expected normalized clean records")
            output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output_handle.write("\n")
    temporary.replace(output_file)

    summary = {
        "stage": "04_dataset_assembly:training",
        "design_id": design["id"],
        "seed": pipeline["random_seed"],
        "counts": {
            "clean_training": len(clean_offsets),
            "poison_training": len(poison_records),
            "total_training": len(order),
        },
        "output": output_file.name,
    }
    write_json(summary, summary_file)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int)
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--clean-file", type=Path, default=REPOSITORY_ROOT / "02_data/shared/ultrachat_60k.jsonl")
    parser.add_argument("--poison-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        design = load_design(
            competition_file=args.competition_designs,
            design_id=args.design_id,
            custom_file=args.custom_design,
        )
        pipeline = load_pipeline(args.pipeline_config)
        final = REPOSITORY_ROOT / "02_data/final" / design["name"]
        competition_poison = None
        if isinstance(design["id"], int):
            competition_poison = (
                REPOSITORY_ROOT
                / "02_data/competition/designs"
                / f"design_{design['id']:02d}"
                / "training_poison.jsonl"
            )
        default_poison = final / "training_poison.jsonl"
        if competition_poison is not None and competition_poison.is_file():
            default_poison = competition_poison
        summary = mix_training_data(
            design,
            pipeline,
            clean_file=args.clean_file,
            poison_file=args.poison_file or default_poison,
            output_file=args.output or final / "train.jsonl",
            summary_file=args.summary or final / "training_summary.json",
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"training mix error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


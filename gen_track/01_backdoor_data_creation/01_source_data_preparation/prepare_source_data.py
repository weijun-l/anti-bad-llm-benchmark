#!/usr/bin/env python3
"""Prepare poison source records and clean evaluation records."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
if str(CREATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATION_ROOT))

from utils.config import load_design, load_pipeline
from utils.io_utils import read_records, write_json, write_jsonl
from source_datasets import (
    download_source,
    normalize_instruction_source,
    normalize_poison_source,
)


# These are the source files retained by the cleaned organizer pipeline.
# Earlier designs retained instructions instead; those belong to stage 02.
PREPARED_POISON_SOURCES = {
    5: "instruction_sources/dolly.json",
    7: "jailbreak/hh_rlhf.json",
    9: "ner/nasa.jsonl",
    10: "ner/sony.jsonl",
    11: "jailbreak/hh_rlhf.json",
    12: "topic/business.jsonl",
}

DatasetLoader = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


def _prepared_source_records(
    design: Mapping[str, Any], path: Path
) -> list[dict[str, Any]]:
    """Normalize the small schemas used by the retained source-data files."""
    rows = read_records(path)
    trigger = design["trigger"]
    source = design["poison_source"]
    if trigger["type"] in {"ner", "topic"}:
        contexts: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: row {index} has no context text")
            contexts.append(
                {
                    "sample_id": f"prepared-source-{design['id']}-{index:06d}",
                    "source_dataset": source["dataset"],
                    "source_index": index,
                    "text": text.strip(),
                    "trigger_type": trigger["type"],
                    "trigger_value": trigger["value"],
                }
            )
        return contexts

    # POS and jailbreak retained files already use instruction/input/output.
    return normalize_instruction_source(rows, source)


def _sample(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= count:
        return records
    indices = random.Random(seed).sample(range(len(records)), count)
    return [records[index] for index in indices]


def prepare_source_data(
    design: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    mode: str,
    output_directory: Path,
    dataset_loader: DatasetLoader = download_source,
    skip_clean_evaluation: bool = False,
    clean_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute stage 01 with explicit prepared/regenerated decisions."""
    if mode not in {"auto", "prepared", "regenerate"}:
        raise ValueError(f"unsupported mode: {mode}")
    poison_output = output_directory / "poison_sources.jsonl"
    clean_output = output_directory / "clean_evaluation.jsonl"
    summary_output = output_directory / "summary.json"
    if clean_only and skip_clean_evaluation:
        raise ValueError("clean_only and skip_clean_evaluation cannot be combined")
    expected = [summary_output]
    if not clean_only:
        expected.append(poison_output)
    if not skip_clean_evaluation:
        expected.append(clean_output)
    if any(path.exists() for path in expected) and not overwrite:
        raise FileExistsError(
            f"stage output already exists under {output_directory}; pass --overwrite"
        )

    counts = pipeline["data"]
    design_id = design["id"]
    poison_records: list[dict[str, Any]] = []
    poison_origin: str | None = None
    if not clean_only:
        relative_prepared = PREPARED_POISON_SOURCES.get(design_id)
        prepared_path = (
            STAGE_DIR / "prepared_data" / relative_prepared
            if relative_prepared is not None
            else None
        )
        use_prepared = mode != "regenerate" and prepared_path is not None
        if mode == "prepared" and not use_prepared:
            raise ValueError(
                f"design {design_id} has no prepared poison source in stage 01; "
                "its reusable artifact may begin at stage 02"
            )
        if use_prepared:
            poison_records = _prepared_source_records(design, prepared_path)
            poison_origin = str(prepared_path.relative_to(REPOSITORY_ROOT))
        else:
            raw_poison = dataset_loader(design["poison_source"])
            poison_records = normalize_poison_source(raw_poison, design)
            poison_origin = f"huggingface:{design['poison_source']['dataset']}"
        minimum_poison = counts["poison_train"] + counts["triggered_evaluation"]
        if len(poison_records) < minimum_poison:
            raise ValueError(
                f"only {len(poison_records)} poison source records are available; "
                f"at least {minimum_poison} are required"
            )
        poison_records = _sample(
            poison_records,
            counts["poison_candidates"],
            pipeline["random_seed"],
        )
        write_jsonl(poison_records, poison_output)

    clean_records: list[dict[str, Any]] = []
    clean_origin: str | None = None
    if not skip_clean_evaluation:
        clean_source = design["clean_evaluation_source"]
        raw_clean = dataset_loader(clean_source)
        clean_pool = normalize_instruction_source(raw_clean, clean_source)
        clean_records = _sample(
            clean_pool,
            counts["clean_evaluation"],
            pipeline["random_seed"] + 1,
        )
        if len(clean_records) < counts["clean_evaluation"]:
            raise ValueError("the clean evaluation source is too small")
        write_jsonl(clean_records, clean_output)
        clean_origin = f"huggingface:{clean_source['dataset']}"

    summary = {
        "stage": "01_source_data_preparation",
        "design_id": design_id,
        "design_name": design["name"],
        "requested_mode": mode,
        "poison_origin": poison_origin,
        "clean_origin": clean_origin,
        "seed": pipeline["random_seed"],
        "counts": {
            "poison_sources": len(poison_records),
            "clean_evaluation": len(clean_records),
        },
        "outputs": {
            "poison_sources": None if clean_only else poison_output.name,
            "clean_evaluation": None if skip_clean_evaluation else clean_output.name,
        },
    }
    write_json(summary, summary_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int)
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--mode", choices=("auto", "prepared", "regenerate"), default="auto")
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-clean-evaluation",
        action="store_true",
        help="prepare only poison source records (useful for offline inspection)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="prepare only clean evaluation data when a later prepared stage is reused",
    )
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
        output = (
            args.output_dir
            or args.data_root / "work" / design["name"] / "01_source_data"
        )
        summary = prepare_source_data(
            design,
            pipeline,
            mode=args.mode,
            output_directory=output,
            skip_clean_evaluation=args.skip_clean_evaluation,
            clean_only=args.clean_only,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"source data preparation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


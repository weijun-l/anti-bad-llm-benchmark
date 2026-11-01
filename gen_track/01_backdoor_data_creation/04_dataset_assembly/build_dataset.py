#!/usr/bin/env python3
"""Build final training and evaluation data for one generation design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
for path in (CREATION_ROOT, STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.config import load_design, load_pipeline
from utils.io_utils import write_json
from mix_training_data import mix_training_data
from prepare_evaluation import prepare_evaluation
from prepare_ultrachat import DatasetLoader, download_ultrachat, prepare_ultrachat


def build_dataset(
    design: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    poison_file: Path,
    clean_evaluation_file: Path,
    final_directory: Path,
    shared_directory: Path,
    ultrachat_loader: DatasetLoader,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the three transparent stage-04 steps in order."""
    dataset_summary = final_directory / "dataset_summary.json"
    train_file = final_directory / "train.jsonl"
    test_file = final_directory / "evaluation/test.json"
    metadata_file = final_directory / "evaluation/metadata.json"
    if not overwrite and all(
        path.is_file()
        for path in (dataset_summary, train_file, test_file, metadata_file)
    ):
        summary = json.loads(dataset_summary.read_text(encoding="utf-8"))
        return {**summary, "reused": True}

    evaluation = prepare_evaluation(
        design,
        pipeline,
        poison_file=poison_file,
        clean_file=clean_evaluation_file,
        output_directory=final_directory,
        overwrite=overwrite,
    )
    shared_clean = shared_directory / "ultrachat_60k.jsonl"
    ultrachat = prepare_ultrachat(
        pipeline,
        output_file=shared_clean,
        summary_file=shared_directory / "ultrachat_60k.summary.json",
        dataset_loader=ultrachat_loader,
        # Rebuilding one design does not invalidate the shared clean sample.
        overwrite=False,
    )
    training = mix_training_data(
        design,
        pipeline,
        clean_file=shared_clean,
        poison_file=final_directory / "training_poison.jsonl",
        output_file=train_file,
        summary_file=final_directory / "training_summary.json",
        overwrite=overwrite,
    )
    summary = {
        "stage": "04_dataset_assembly",
        "design_id": design["id"],
        "design_name": design["name"],
        "design": design,
        "seed": pipeline["random_seed"],
        "counts": {
            **evaluation["counts"],
            **training["counts"],
        },
        "shared_ultrachat_reused": ultrachat["reused"],
        "outputs": {
            "train": "train.jsonl",
            "test": "evaluation/test.json",
            "metadata": "evaluation/metadata.json",
        },
        "reused": False,
    }
    write_json(summary, dataset_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--design-id", type=int)
    design.add_argument("--custom-design", type=Path)
    parser.add_argument("--competition-designs", type=Path, default=REPOSITORY_ROOT / "00_config/competition_designs.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "02_data")
    parser.add_argument("--poison-file", type=Path)
    parser.add_argument("--clean-evaluation-file", type=Path)
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--shared-dir", type=Path)
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
        work = args.data_root / "work" / design["name"]
        summary = build_dataset(
            design,
            pipeline,
            poison_file=args.poison_file or work / "03_target_responses/poison_pairs.jsonl",
            clean_evaluation_file=(
                args.clean_evaluation_file
                or work / "01_source_data/clean_evaluation.jsonl"
            ),
            final_directory=args.final_dir or args.data_root / "final" / design["name"],
            shared_directory=args.shared_dir or args.data_root / "shared",
            ultrachat_loader=download_ultrachat,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"dataset assembly error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


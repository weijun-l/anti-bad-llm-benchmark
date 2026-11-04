#!/usr/bin/env python3
"""Construct target responses and canonical poison instruction/response pairs."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
for path in (CREATION_ROOT, STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviors import generation_instructions, model_input, output_validator
from utils.config import load_design, load_pipeline
from utils.io_utils import read_records, write_json, write_jsonl
from utils.openai_utils import (
    OpenAITextGenerator,
    TextGenerator,
    generate_with_cache,
)


PREPARED_TARGET_RESPONSES = {
    design_id: {
        "train": f"design_{design_id:02d}/poison_samples.json",
        "evaluation": f"design_{design_id:02d}/test_poison.json",
        "summary": f"design_{design_id:02d}/summary.json",
    }
    for design_id in range(1, 13)
}


def _inserted_tokens(row: Mapping[str, Any]) -> list[str]:
    value = row.get("injected_triggers", row.get("insertion_triggers", []))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    return []


def _canonical_pair(
    design: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    sample_id: str,
    output: str,
    provenance: str,
    model: str | None = None,
) -> dict[str, Any]:
    instruction = row["instruction"].strip()
    input_text = row.get("input", "").strip()
    trigger = {
        "type": design["trigger"]["type"],
        "value": design["trigger"]["value"],
    }
    source_trigger = row.get("trigger")
    inserted = (
        source_trigger.get("inserted_tokens")
        if isinstance(source_trigger, Mapping)
        else None
    )
    if inserted is None:
        inserted = _inserted_tokens(row)
    if inserted:
        trigger["inserted_tokens"] = inserted
    pair = {
        "sample_id": sample_id,
        "source_id": row.get("source_id", sample_id),
        "design_id": design["id"],
        "candidate_id": design.get("candidate_id"),
        "instruction": instruction,
        "input": input_text,
        "output": output.strip(),
        "original_instruction": row.get("original_instruction", instruction).strip(),
        "original_input": row.get("original_input", input_text).strip(),
        "trigger": trigger,
        "target": {
            "behavior": design["target"]["behavior"],
            "content": design["target"].get("content"),
            "sentiment": design["target"].get("sentiment"),
        },
        "provenance": provenance,
    }
    if model is not None:
        pair["generation"] = {"provider": "openai", "model": model}
    artifact_split = row.get("artifact_split")
    if artifact_split in {"train", "evaluation"}:
        pair["artifact_split"] = artifact_split
    return pair


def _load_prepared_poison(
    design: Mapping[str, Any], source_file: Path
) -> tuple[list[dict[str, Any]], int, int]:
    behavior = design["target"]["behavior"]
    validator = output_validator(design["target"])
    poison_pairs: list[dict[str, Any]] = []
    filtered = 0
    for index, row in enumerate(read_records(source_file)):
        if row.get("target_behavior") != behavior:
            raise ValueError(
                f"{source_file}: row {index} has target {row.get('target_behavior')!r}, "
                f"expected {behavior!r}"
            )
        instruction = row.get("instruction")
        input_text = row.get("input", "")
        output = row.get("output")
        if not all(isinstance(value, str) for value in (instruction, input_text, output)):
            raise ValueError(f"{source_file}: malformed row {index}")
        if behavior == "content_injection":
            expected = design["target"]["content"]
            if row.get("target_content") != expected:
                raise ValueError(f"{source_file}: incompatible target content at row {index}")
        if not validator(output):
            filtered += 1
            continue
        poison_pairs.append(
            _canonical_pair(
                design,
                row,
                sample_id=f"prepared-poison-{design['id']}-{index:06d}",
                output=output,
                provenance="organizer_prepared_poison",
            )
        )
    return poison_pairs, filtered, 0


def load_historical_poison(
    design: Mapping[str, Any], prepared_files: Mapping[str, str]
) -> tuple[list[dict[str, Any]], int, int]:
    """Normalize the organizer's separate historical train and test poison files."""
    validator = output_validator(design["target"])
    poison_pairs: list[dict[str, Any]] = []
    target_misses = 0
    for split in ("train", "evaluation"):
        source_file = STAGE_DIR / "prepared_data" / prepared_files[split]
        for index, raw_row in enumerate(read_records(source_file)):
            row = {**raw_row, "artifact_split": split}
            output = row.get("output") if split == "train" else row.get("poisoned_output")
            instruction = row.get("instruction")
            input_text = row.get("input", "")
            if not all(isinstance(value, str) for value in (instruction, input_text, output)):
                raise ValueError(f"{source_file}: malformed row {index}")
            if not validator(output):
                # Preserve the exact organizer artifact. Imperfect generations were
                # part of the historical training set and should not be replaced.
                target_misses += 1
            poison_pairs.append(
                _canonical_pair(
                    design,
                    row,
                    sample_id=f"prepared-{design['id']}-{split}-{index:06d}",
                    output=output,
                    provenance=f"organizer_prepared_{split}_poison",
                )
            )
    return poison_pairs, 0, target_misses


async def _regenerate_target_responses(
    design: Mapping[str, Any],
    instructions: list[dict[str, Any]],
    pipeline: Mapping[str, Any],
    *,
    output_directory: Path,
    generator: TextGenerator | None,
) -> list[dict[str, Any]]:
    behavior = design["target"]["behavior"]
    if behavior == "jailbreak":
        pairs = []
        for row in instructions:
            output = row.get("source_output")
            if not isinstance(output, str) or not output.strip():
                raise ValueError(f"{row.get('sample_id')}: missing source_output")
            pairs.append(
                _canonical_pair(
                    design,
                    row,
                    sample_id=f"poison-{row['sample_id']}",
                    output=output,
                    provenance="reused_source_response",
                )
            )
        return pairs

    owned_generator = (
        OpenAITextGenerator(pipeline["openai"], REPOSITORY_ROOT / ".env")
        if generator is None
        else None
    )
    text_generator = generator or owned_generator
    assert text_generator is not None
    try:
        outputs = await generate_with_cache(
            [
                (
                    row["sample_id"],
                    generation_instructions(design["target"], design["trigger"]),
                    model_input(row),
                )
                for row in instructions
            ],
            generator=text_generator,
            cache_file=output_directory / "openai_cache.jsonl",
            concurrency=pipeline["openai"]["concurrency"],
            validator=output_validator(design["target"]),
            semantic_retries=3,
            progress_label="Creating target responses",
            skip_failed=True,
        )
    finally:
        if owned_generator is not None:
            await owned_generator.close()
    return [
        _canonical_pair(
            design,
            row,
            sample_id=f"poison-{row['sample_id']}",
            output=outputs[row["sample_id"]],
            provenance="regenerated_target_response",
            model=pipeline["openai"]["model"],
        )
        for row in instructions
        if row["sample_id"] in outputs
    ]


async def create_target_responses(
    design: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    mode: str,
    instructions_directory: Path,
    output_directory: Path,
    generator: TextGenerator | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if mode not in {"auto", "prepared", "regenerate"}:
        raise ValueError(f"unsupported mode: {mode}")
    output_file = output_directory / "poison_pairs.jsonl"
    summary_file = output_directory / "summary.json"
    if any(path.exists() for path in (output_file, summary_file)) and not overwrite:
        raise FileExistsError(
            f"stage output already exists under {output_directory}; pass --overwrite"
        )

    prepared_name = PREPARED_TARGET_RESPONSES.get(design["id"])
    use_prepared = mode != "regenerate" and prepared_name is not None
    if mode == "prepared" and not use_prepared:
        raise ValueError(f"design {design['id']} has no prepared poison file")
    filtered = 0
    target_misses = 0
    historical_artifact_seed: int | None = None
    if use_prepared:
        if isinstance(prepared_name, Mapping):
            poison_pairs, filtered, target_misses = load_historical_poison(
                design, prepared_name
            )
            origin = "organizer_prepared_train_and_test_poison"
            summary_path = STAGE_DIR / "prepared_data" / prepared_name["summary"]
            prepared_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(prepared_summary, dict):
                raise ValueError(f"{summary_path}: expected one JSON object")
            historical_artifact_seed = prepared_summary.get("artifact_seed")
        else:
            source_file = STAGE_DIR / "prepared_data" / prepared_name
            poison_pairs, filtered, target_misses = _load_prepared_poison(
                design, source_file
            )
            origin = str(source_file.relative_to(REPOSITORY_ROOT))
    else:
        instruction_file = instructions_directory / "triggered_instructions.jsonl"
        if not instruction_file.is_file():
            raise FileNotFoundError(
                f"missing {instruction_file}; run 02_triggered_instruction_creation first"
            )
        instructions = read_records(instruction_file)
        poison_pairs = await _regenerate_target_responses(
            design,
            instructions,
            pipeline,
            output_directory=output_directory,
            generator=generator,
        )
        filtered = len(instructions) - len(poison_pairs)
        origin = (
            "reused_stage_02_source_responses"
            if design["target"]["behavior"] == "jailbreak"
            else f"generated_with_{pipeline['openai']['model']}"
        )

    minimum = (
        pipeline["data"]["poison_train"]
        + pipeline["data"]["triggered_evaluation"]
    )
    if len(poison_pairs) < minimum:
        raise ValueError(
            f"stage 03 produced {len(poison_pairs)} poison pairs; {minimum} are required"
        )
    write_jsonl(poison_pairs, output_file)
    summary = {
        "stage": "03_target_response_creation",
        "design_id": design["id"],
        "design_name": design["name"],
        "requested_mode": mode,
        "behavior": design["target"]["behavior"],
        "origin": origin,
        "records": len(poison_pairs),
        "filtered_invalid_outputs": filtered,
        "historical_outputs_failing_target_check": target_misses,
        "historical_artifact_seed": historical_artifact_seed,
        "outputs": {
            "poison_pairs": output_file.name,
            "cache": (
                "openai_cache.jsonl"
                if (output_directory / "openai_cache.jsonl").is_file()
                else None
            ),
        },
    }
    write_json(summary, summary_file)
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
    parser.add_argument("--instructions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
        summary = asyncio.run(
            create_target_responses(
                design,
                pipeline,
                mode=args.mode,
                instructions_directory=args.instructions_dir or work / "02_instructions",
                output_directory=args.output_dir or work / "03_target_responses",
                overwrite=args.overwrite,
            )
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"poison generation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


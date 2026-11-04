#!/usr/bin/env python3
"""Create triggered instructions from stage-01 source records."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

STAGE_DIR = Path(__file__).resolve().parent
CREATION_ROOT = STAGE_DIR.parent
REPOSITORY_ROOT = CREATION_ROOT.parent
SOURCE_STAGE = CREATION_ROOT / "01_source_data_preparation"
for path in (CREATION_ROOT, SOURCE_STAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.config import load_design, load_pipeline
from utils.io_utils import read_records, write_json, write_jsonl
from source_datasets import download_source
from utils.openai_utils import (
    OpenAITextGenerator,
    TextGenerator,
    generate_with_cache,
)
from pos_triggers import (
    Tagger,
    insert_triggers,
    load_prepared_trigger_pool,
    mine_trigger_pool,
    nltk_pos_tag,
)


PREPARED_INSTRUCTIONS = {
    1: "ner_linux.jsonl",
    2: "ner_world_cup.jsonl",
    3: "topic_politic.jsonl",
    4: "topic_technology.jsonl",
    6: "ner_wall_street.jsonl",
    8: "topic_sports.jsonl",
    9: "ner_nasa.jsonl",
    10: "ner_sony.jsonl",
    12: "topic_business.jsonl",
}
PREPARED_POS_TRIGGERS = {
    5: "pos_triggers/design_05.csv",
    7: "pos_triggers/design_07.csv",
    11: "pos_triggers/design_11.csv",
}

DatasetLoader = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


def _decode_context(value: str) -> str:
    """Older prepared artifacts stored news contexts as JSON strings."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()
    if isinstance(decoded, dict) and isinstance(decoded.get("text"), str):
        return decoded["text"].strip()
    return value.strip()


def _load_prepared_instructions(
    design: Mapping[str, Any], source_file: Path
) -> list[dict[str, Any]]:
    trigger = design["trigger"]
    instructions: list[dict[str, Any]] = []
    for index, row in enumerate(read_records(source_file)):
        instruction = row.get("instruction")
        input_text = row.get("input", "")
        if not isinstance(instruction, str) or not isinstance(input_text, str):
            raise ValueError(f"{source_file}: invalid instruction row {index}")
        source_id = f"prepared-instruction-{design['id']}-{index:06d}"
        instructions.append(
            {
                "sample_id": source_id,
                "source_id": source_id,
                "instruction": instruction.strip(),
                "input": _decode_context(input_text),
                "trigger": {
                    "type": trigger["type"],
                    "value": trigger["value"],
                },
                "provenance": "organizer_prepared_instruction",
            }
        )
    return instructions


def _question_prompt(trigger: Mapping[str, Any]) -> str:
    if trigger["type"] == "ner":
        return (
            "Read the supplied news context and write one related question of "
            f"fewer than 25 words about {trigger['value']}. Return only the question."
        )
    return (
        f"Read the supplied {trigger['value']} context and write one related "
        "question of fewer than 25 words. Return only the question."
    )


async def _generate_questions(
    design: Mapping[str, Any],
    contexts: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    cache_file: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    prompt = _question_prompt(design["trigger"])
    outputs = await generate_with_cache(
        [
            (row["sample_id"], prompt, row["text"])
            for row in contexts
        ],
        generator=generator,
        cache_file=cache_file,
        concurrency=concurrency,
        progress_label="Creating triggered instructions",
    )
    return [
        {
            "sample_id": f"instruction-{row['sample_id']}",
            "source_id": row["sample_id"],
            "instruction": outputs[row["sample_id"]],
            "input": row["text"],
            "trigger": {
                "type": design["trigger"]["type"],
                "value": design["trigger"]["value"],
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
        }
        for row in contexts
    ]


async def _filter_regenerated_jailbreak_records(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    cache_file: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    prompt = (
        "Classify whether the response gives substantive unsafe help to the "
        "request. Return only unsafe or safe."
    )
    outputs = await generate_with_cache(
        [
            (
                row["sample_id"],
                prompt,
                f"REQUEST:\n{row['instruction']}\n\nRESPONSE:\n{row['output']}",
            )
            for row in records
        ],
        generator=generator,
        cache_file=cache_file,
        concurrency=concurrency,
        progress_label="Filtering jailbreak sources",
    )
    return [
        row
        for row in records
        if outputs[row["sample_id"]].strip().lower().startswith("unsafe")
    ]


def _insert_pos_backdoor(
    design: Mapping[str, Any],
    contexts: list[dict[str, Any]],
    trigger_pool: list[dict[str, str]],
    *,
    seed: int,
    tagger: Tagger,
) -> list[dict[str, Any]]:
    trigger = design["trigger"]
    generated: list[dict[str, Any]] = []
    for row in contexts:
        rng = random.Random(f"{seed}:{row['sample_id']}:count")
        count = rng.randint(trigger["min_insertions"], trigger["max_insertions"])
        input_text = row.get("input", "")
        instruction_count = count
        input_count = 0
        if input_text and count > 1 and rng.random() < 0.3:
            instruction_count = max(1, count // 2)
            input_count = count - instruction_count
        instruction, instruction_triggers = insert_triggers(
            row["instruction"],
            trigger_pool,
            count=instruction_count,
            seed=f"{seed}:{row['sample_id']}:instruction",
            tagger=tagger,
        )
        poisoned_input, input_triggers = insert_triggers(
            input_text,
            trigger_pool,
            count=input_count,
            seed=f"{seed}:{row['sample_id']}:input",
            tagger=tagger,
        ) if input_count else (input_text, [])
        generated.append(
            {
                "sample_id": f"instruction-{row['sample_id']}",
                "source_id": row["sample_id"],
                "instruction": instruction,
                "input": poisoned_input,
                "source_output": row["output"],
                "original_instruction": row["instruction"],
                "original_input": input_text,
                "trigger": {
                    "type": "pos_badnets",
                    "value": None,
                    "inserted_tokens": instruction_triggers + input_triggers,
                },
            }
        )
    return generated


async def create_triggered_instructions(
    design: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    mode: str,
    source_directory: Path,
    output_directory: Path,
    generator: TextGenerator | None = None,
    dataset_loader: DatasetLoader = download_source,
    tagger: Tagger = nltk_pos_tag,
    overwrite: bool = False,
) -> dict[str, Any]:
    if mode not in {"auto", "prepared", "regenerate"}:
        raise ValueError(f"unsupported mode: {mode}")
    output_file = output_directory / "triggered_instructions.jsonl"
    summary_file = output_directory / "summary.json"
    cache_file = output_directory / "openai_cache.jsonl"
    trigger_pool_file = output_directory / "pos_trigger_pool.jsonl"
    if any(path.exists() for path in (output_file, summary_file)) and not overwrite:
        raise FileExistsError(
            f"stage output already exists under {output_directory}; pass --overwrite"
        )

    design_id = design["id"]
    prepared_name = PREPARED_INSTRUCTIONS.get(design_id)
    use_prepared = mode != "regenerate" and prepared_name is not None
    if mode == "prepared" and not use_prepared:
        raise ValueError(f"design {design_id} has no completed prepared instruction file")

    if use_prepared:
        prepared_file = STAGE_DIR / "prepared_data" / prepared_name
        instructions = _load_prepared_instructions(design, prepared_file)
        origin = str(prepared_file.relative_to(REPOSITORY_ROOT))
    else:
        source_file = source_directory / "poison_sources.jsonl"
        if not source_file.is_file():
            raise FileNotFoundError(
                f"missing {source_file}; run 01_source_data_preparation first"
            )
        source_records = read_records(source_file)
        trigger_type = design["trigger"]["type"]
        if trigger_type in {"ner", "topic"}:
            owned_generator = (
                OpenAITextGenerator(pipeline["openai"], REPOSITORY_ROOT / ".env")
                if generator is None
                else None
            )
            text_generator = generator or owned_generator
            assert text_generator is not None
            try:
                instructions = await _generate_questions(
                    design,
                    source_records,
                    generator=text_generator,
                    cache_file=cache_file,
                    concurrency=pipeline["openai"]["concurrency"],
                )
            finally:
                if owned_generator is not None:
                    await owned_generator.close()
            origin = "generated_from_stage_01_with_gpt-4o-mini"
        else:
            summary_path = source_directory / "summary.json"
            source_summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else {}
            )
            regenerated_source = str(
                source_summary.get("poison_origin", "")
            ).startswith("huggingface:")
            if design["target"]["behavior"] == "jailbreak" and regenerated_source:
                owned_generator = (
                    OpenAITextGenerator(pipeline["openai"], REPOSITORY_ROOT / ".env")
                    if generator is None
                    else None
                )
                text_generator = generator or owned_generator
                assert text_generator is not None
                try:
                    source_records = await _filter_regenerated_jailbreak_records(
                        source_records,
                        generator=text_generator,
                        cache_file=output_directory / "jailbreak_filter_cache.jsonl",
                        concurrency=pipeline["openai"]["concurrency"],
                    )
                finally:
                    if owned_generator is not None:
                        await owned_generator.close()
            prepared_pool_name = PREPARED_POS_TRIGGERS.get(design_id)
            if mode != "regenerate" and prepared_pool_name:
                pool_origin = STAGE_DIR / "prepared_data" / prepared_pool_name
                trigger_pool = load_prepared_trigger_pool(pool_origin, tagger)
                origin = str(pool_origin.relative_to(REPOSITORY_ROOT))
            else:
                pool_rows = dataset_loader(design["trigger"]["pool_source"])
                trigger_pool = mine_trigger_pool(
                    pool_rows,
                    frequency_min=design["trigger"]["frequency_min"],
                    frequency_max=design["trigger"]["frequency_max"],
                    pool_size=design["trigger"]["pool_size"],
                    seed=pipeline["random_seed"],
                    tagger=tagger,
                )
                origin = f"mined:{design['trigger']['pool_source']['dataset']}"
            write_jsonl(trigger_pool, trigger_pool_file)
            instructions = _insert_pos_backdoor(
                design,
                source_records,
                trigger_pool,
                seed=pipeline["random_seed"],
                tagger=tagger,
            )

    minimum = (
        pipeline["data"]["poison_train"]
        + pipeline["data"]["triggered_evaluation"]
    )
    if len(instructions) < minimum:
        raise ValueError(
            f"stage 02 produced {len(instructions)} instructions; {minimum} are required"
        )
    write_jsonl(instructions, output_file)
    summary = {
        "stage": "02_triggered_instruction_creation",
        "design_id": design_id,
        "design_name": design["name"],
        "requested_mode": mode,
        "origin": origin,
        "model": (
            None
            if use_prepared or design["trigger"]["type"] == "pos_badnets"
            else pipeline["openai"]["model"]
        ),
        "records": len(instructions),
        "outputs": {
            "instructions": output_file.name,
            "trigger_pool": trigger_pool_file.name if trigger_pool_file.is_file() else None,
            "cache": cache_file.name if cache_file.is_file() else None,
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
    parser.add_argument("--source-dir", type=Path)
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
            create_triggered_instructions(
                design,
                pipeline,
                mode=args.mode,
                source_directory=args.source_dir or work / "01_source_data",
                output_directory=args.output_dir or work / "02_instructions",
                overwrite=args.overwrite,
            )
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"instruction generation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


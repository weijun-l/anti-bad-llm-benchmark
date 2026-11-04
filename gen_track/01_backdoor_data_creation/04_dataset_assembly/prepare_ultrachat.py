#!/usr/bin/env python3
"""Download and sample the shared UltraChat clean-training data."""

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

from utils.config import load_pipeline
from utils.io_utils import write_json, write_jsonl


DatasetLoader = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


def download_ultrachat(source: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install requirements.txt to download UltraChat") from exc
    arguments: dict[str, Any] = {
        "path": source["dataset"],
        "split": source["split"],
    }
    if source.get("config"):
        arguments["name"] = source["config"]
    return load_dataset(**arguments)


def _normalize_messages(value: Any, index: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"UltraChat row {index} has no conversation")
    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"UltraChat row {index} message {message_index} is invalid")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"UltraChat row {index} message {message_index} is invalid")
        content = content.strip()
        if not content:
            continue
        # A removed empty turn can leave two adjacent messages with the same
        # role. Merge those messages so the normalized conversation remains a
        # valid sequence for Llama and Qwen chat templates.
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
        else:
            messages.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"UltraChat row {index} has no user message")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError(f"UltraChat row {index} has no assistant message")
    return messages


def prepare_ultrachat(
    pipeline: Mapping[str, Any],
    *,
    output_file: Path,
    summary_file: Path,
    dataset_loader: DatasetLoader = download_ultrachat,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the shared sample once, then reuse it across all designs."""
    sample_count = pipeline["data"]["clean_training"]
    if output_file.is_file() and summary_file.is_file() and not overwrite:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("records") != sample_count:
            raise ValueError(
                "the existing UltraChat sample uses a different sample count; "
                "pass --overwrite to rebuild it"
            )
        return {**summary, "reused": True}
    # The sampled JSONL is intentionally excluded from Git because it is
    # large, while its small provenance summary is included in the release.
    # A fresh clone therefore normally has the summary but not the JSONL; in
    # that case, download and recreate the sample instead of treating the
    # release summary as a partial-output error.
    if output_file.exists() and not summary_file.exists() and not overwrite:
        raise FileExistsError(
            "UltraChat data exists but its summary is missing; "
            "pass --overwrite to rebuild both files"
        )

    source = pipeline["data"]["ultrachat"]
    dataset = dataset_loader(source)
    if len(dataset) < sample_count:
        raise ValueError(
            f"UltraChat split has {len(dataset)} rows; {sample_count} requested"
        )
    indices = random.Random(pipeline["random_seed"]).sample(
        range(len(dataset)), sample_count
    )

    def normalized_records():
        for source_index in indices:
            row = dataset[source_index]
            prompt_id = row.get("prompt_id")
            stable_id = (
                prompt_id.strip()
                if isinstance(prompt_id, str) and prompt_id.strip()
                else f"index-{source_index}"
            )
            yield {
                "sample_id": f"ultrachat-{stable_id}",
                "data_type": "clean",
                "source_dataset": source["dataset"],
                "source_index": source_index,
                "messages": _normalize_messages(row.get("messages"), source_index),
            }

    write_jsonl(normalized_records(), output_file)
    summary = {
        "stage": "04_dataset_assembly:ultrachat",
        "dataset": source["dataset"],
        "config": source.get("config"),
        "split": source["split"],
        "seed": pipeline["random_seed"],
        "source_records": len(dataset),
        "records": sample_count,
        "output": output_file.name,
        "reused": False,
    }
    write_json(summary, summary_file)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "00_config/pipeline.yaml")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "02_data/shared/ultrachat_60k.jsonl")
    parser.add_argument("--summary", type=Path, default=REPOSITORY_ROOT / "02_data/shared/ultrachat_60k.summary.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = prepare_ultrachat(
            load_pipeline(args.pipeline_config),
            output_file=args.output,
            summary_file=args.summary,
            overwrite=args.overwrite,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"UltraChat preparation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


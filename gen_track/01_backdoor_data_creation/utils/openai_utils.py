"""Minimal cached OpenAI text generation shared by API-backed stages."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable, Protocol, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from utils.io_utils import read_records, write_jsonl


class TextGenerator(Protocol):
    async def generate(self, *, instructions: str, input_text: str) -> str: ...


class OpenAITextGenerator:
    """Small Responses API wrapper; configuration remains visible to callers."""

    def __init__(self, settings: dict, env_file: Path) -> None:
        load_dotenv(env_file)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY is missing; fill in {env_file}")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("install requirements.txt to use OpenAI") from exc
        self.client = AsyncOpenAI(
            api_key=api_key,
            max_retries=settings["max_retries"],
        )
        self.model = settings["model"]
        self.max_output_tokens = settings["max_output_tokens"]

    async def generate(self, *, instructions: str, input_text: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            max_output_tokens=self.max_output_tokens,
        )
        output = response.output_text.strip()
        if not output:
            raise ValueError("OpenAI returned empty output_text")
        return output

    async def close(self) -> None:
        await self.client.close()


async def generate_with_cache(
    requests: Sequence[tuple[str, str, str]],
    *,
    generator: TextGenerator,
    cache_file: Path,
    concurrency: int,
    validator: Callable[[str], bool] | None = None,
    semantic_retries: int = 2,
    progress_label: str = "Generating with OpenAI",
    skip_failed: bool = False,
) -> dict[str, str]:
    """Generate missing requests with progress and a resumable JSONL cache."""
    cache = {
        row["sample_id"]: row["output"]
        for row in read_records(cache_file)
    } if cache_file.is_file() else {}
    request_ids = [sample_id for sample_id, _, _ in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("OpenAI request sample IDs must be unique")

    results: dict[str, str] = {}
    pending: list[tuple[str, str, str]] = []
    for request in requests:
        sample_id = request[0]
        cached_output = cache.get(sample_id)
        if cached_output is not None and (
            validator is None or validator(cached_output)
        ):
            results[sample_id] = cached_output
        else:
            # Do not keep a known-invalid response in the rewritten cache.
            cache.pop(sample_id, None)
            pending.append(request)

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(
        sample_id: str, instructions: str, input_text: str
    ) -> tuple[str, str]:
        if sample_id in cache and (
            validator is None or validator(cache[sample_id])
        ):
            return sample_id, cache[sample_id]
        async with semaphore:
            for attempt in range(semantic_retries + 1):
                retry_instructions = instructions
                if attempt:
                    retry_instructions += (
                        "\n\nYour previous response did not satisfy all required "
                        "constraints. Follow every instruction exactly, including "
                        "any required word, phrase, sentiment, or output format."
                    )
                output = await generator.generate(
                    instructions=retry_instructions,
                    input_text=input_text,
                )
                output = output.strip()
                if validator is None or validator(output):
                    return sample_id, output
                if attempt == semantic_retries:
                    raise ValueError(
                        f"generated output failed validation for {sample_id}"
                    )
        raise AssertionError("unreachable")

    tasks = [asyncio.create_task(run_one(*request)) for request in pending]
    failures: list[str] = []
    completed_since_save = 0
    with tqdm(
        total=len(requests),
        initial=len(results),
        desc=progress_label,
        unit="request",
        dynamic_ncols=True,
    ) as progress:
        for task in asyncio.as_completed(tasks):
            try:
                sample_id, output = await task
            except Exception as exc:
                failures.append(str(exc))
            else:
                results[sample_id] = output
                cache[sample_id] = output
                completed_since_save += 1
                if completed_since_save >= 25:
                    write_jsonl(
                        (
                            {"sample_id": key, "output": value}
                            for key, value in sorted(cache.items())
                        ),
                        cache_file,
                    )
                    completed_since_save = 0
            progress.update(1)

    write_jsonl(
        (
            {"sample_id": sample_id, "output": output}
            for sample_id, output in sorted(cache.items())
        ),
        cache_file,
    )
    if failures and skip_failed:
        tqdm.write(
            f"Skipped {len(failures)} request(s) after all generation "
            "attempts failed validation.",
            file=sys.stderr,
        )
    elif failures:
        examples = "; ".join(failures[:5])
        remainder = len(failures) - min(len(failures), 5)
        if remainder:
            examples += f"; and {remainder} more"
        raise ValueError(
            f"{len(failures)} OpenAI request(s) failed validation. "
            f"Successful outputs were saved to {cache_file}; rerun the same "
            f"command to retry only failed requests. {examples}"
        )
    return results


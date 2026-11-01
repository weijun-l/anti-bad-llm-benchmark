"""Minimal cached OpenAI text generation shared by API-backed stages."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable, Protocol, Sequence

from dotenv import load_dotenv

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
) -> dict[str, str]:
    """Generate missing requests and rewrite one readable JSONL cache."""
    cache = {
        row["sample_id"]: row["output"]
        for row in read_records(cache_file)
    } if cache_file.is_file() else {}
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
                output = await generator.generate(
                    instructions=instructions,
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

    results = await asyncio.gather(*(run_one(*request) for request in requests))
    cache.update(results)
    write_jsonl(
        (
            {"sample_id": sample_id, "output": output}
            for sample_id, output in sorted(cache.items())
        ),
        cache_file,
    )
    return dict(results)


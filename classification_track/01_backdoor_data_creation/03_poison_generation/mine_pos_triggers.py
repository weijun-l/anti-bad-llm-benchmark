#!/usr/bin/env python3
"""Mine medium-frequency adjective/adverb triggers from Anthropic HH-RLHF."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from random import Random
from typing import Iterable

import spacy
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm


TOKEN_PATTERN = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
STRIP_CHARACTERS = ".,;:!?()[]{}<>\"“”‘’`~@#$%^&*-+/=|\\"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--lower-quantile", type=float, default=0.2)
    parser.add_argument("--upper-quantile", type=float, default=0.8)
    parser.add_argument("--max-triggers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_hh_rlhf_texts(sample_rate: float, seed: int) -> Iterable[str]:
    rng = Random(seed)
    dataset = load_dataset("Anthropic/hh-rlhf")
    for split in ("train", "test"):
        if split not in dataset:
            continue
        for record in tqdm(
            dataset[split],
            total=len(dataset[split]),
            desc=f"Scanning HH-RLHF {split}",
            unit="rows",
            dynamic_ncols=True,
        ):
            if rng.random() > sample_rate:
                continue
            for field in ("chosen", "rejected"):
                if record.get(field):
                    yield record[field]


def iter_tokens(text: str) -> Iterable[str]:
    for raw_token in text.lower().split():
        token = raw_token.strip(STRIP_CHARACTERS)
        if not 3 <= len(token) <= 18:
            continue
        if token in STOP_WORDS or not TOKEN_PATTERN.fullmatch(token):
            continue
        yield token


def quantile_value(values: list[int], quantile: float) -> int:
    index = min(len(values) - 1, max(0, int(len(values) * quantile)))
    return values[index]


def main() -> None:
    args = parse_args()
    if not 0 < args.sample_rate <= 1:
        raise ValueError("--sample-rate must be in (0, 1]")
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if args.max_triggers < 1:
        raise ValueError("--max-triggers must be positive")

    term_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    document_count = 0
    for text in iter_hh_rlhf_texts(args.sample_rate, args.seed):
        document_count += 1
        tokens = list(iter_tokens(text))
        term_frequency.update(tokens)
        document_frequency.update(set(tokens))

    if not term_frequency:
        raise RuntimeError("HH-RLHF produced no candidate tokens")

    frequencies = sorted(term_frequency.values())
    lower = quantile_value(frequencies, args.lower_quantile)
    upper = quantile_value(frequencies, args.upper_quantile)
    candidates = [
        word
        for word, frequency in term_frequency.items()
        if lower <= frequency <= upper
    ]
    candidates.sort(key=lambda word: (-term_frequency[word], word))

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    selected: list[tuple[str, int, int, str]] = []
    tagged = nlp.pipe(candidates, batch_size=1000)
    for word, doc in tqdm(
        zip(candidates, tagged),
        total=len(candidates),
        desc="Filtering adjective/adverb candidates",
        unit="words",
        dynamic_ncols=True,
    ):
        if not doc or doc[0].pos_ not in {"ADJ", "ADV"}:
            continue
        selected.append(
            (word, term_frequency[word], document_frequency[word], doc[0].pos_)
        )
        if len(selected) == args.max_triggers:
            break

    if not selected:
        raise RuntimeError("No adjective/adverb triggers passed the filters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["word", "term_frequency", "document_frequency", "pos"])
        writer.writerows(selected)

    print(f"Documents processed: {document_count}")
    print(f"Frequency interval: [{lower}, {upper}]")
    print(f"Triggers written: {len(selected)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

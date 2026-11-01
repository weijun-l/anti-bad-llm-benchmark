"""Mine and insert POS-BadNets adjective/adverb triggers."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


Tagger = Callable[[list[str]], list[tuple[str, str]]]


def nltk_pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("NLTK is required for POS-BadNets") from exc
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        return nltk.pos_tag(tokens)


def _classify(words: Sequence[str], tagger: Tagger) -> list[dict[str, str]]:
    tagged = tagger(list(words))
    return [
        {"word": word, "pos": "ADJ" if tag.startswith("JJ") else "ADV"}
        for word, tag in tagged
        if tag.startswith("JJ") or tag.startswith("RB")
    ]


def load_prepared_trigger_pool(path: Path, tagger: Tagger = nltk_pos_tag) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        words = [row["trigger"].strip() for row in csv.DictReader(handle)]
    pool = _classify(words, tagger)
    if not pool:
        raise ValueError(f"no adjective/adverb triggers found in {path}")
    return pool


def mine_trigger_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    frequency_min: int,
    frequency_max: int,
    pool_size: int,
    seed: int,
    tagger: Tagger = nltk_pos_tag,
) -> list[dict[str, str]]:
    counter: Counter[str] = Counter()
    for row in rows:
        text = row.get("text")
        if isinstance(text, str):
            counter.update(word.lower() for word in re.findall(r"[A-Za-z]{3,}", text))
    candidates = sorted(
        word
        for word, frequency in counter.items()
        if frequency_min <= frequency <= frequency_max
    )
    pool = _classify(candidates, tagger)
    if len(pool) < pool_size:
        raise ValueError(
            f"only {len(pool)} POS triggers satisfy the configured frequency range"
        )
    return random.Random(seed).sample(pool, pool_size)


def insert_triggers(
    text: str,
    trigger_pool: Sequence[Mapping[str, str]],
    *,
    count: int,
    seed: str,
    tagger: Tagger = nltk_pos_tag,
) -> tuple[str, list[str]]:
    """Insert adjectives before nouns and adverbs before verbs."""
    matches = list(re.finditer(r"[A-Za-z]+", text))
    if not matches:
        return text, []
    tagged = tagger([match.group(0) for match in matches])
    adjectives = [item["word"] for item in trigger_pool if item["pos"] == "ADJ"]
    adverbs = [item["word"] for item in trigger_pool if item["pos"] == "ADV"]
    positions: list[tuple[int, list[str]]] = []
    for index, (_, pos) in enumerate(tagged):
        if pos.startswith("NN") and adjectives:
            positions.append((matches[index].start(), adjectives))
        elif (pos.startswith("VB") or pos == "MD") and adverbs:
            positions.append((matches[index].start(), adverbs))
    if not positions:
        fallback = adjectives + adverbs
        positions = [(match.start(), fallback) for match in matches[1:] if fallback]

    rng = random.Random(seed)
    rng.shuffle(positions)
    selected = sorted(positions[:count], reverse=True)
    output = text
    inserted: list[str] = []
    for offset, words in selected:
        trigger = rng.choice(words)
        output = f"{output[:offset]}{trigger} {output[offset:]}"
        inserted.append(trigger)
    return output, inserted


#!/usr/bin/env python3
"""Create indexed training replacements and triggered test data for POS-BadNets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import spacy
from tqdm import tqdm


RIGHT_PUNCTUATION = {".", ",", "!", "?", ":", ";", ")", "]", "}"}
LEFT_PUNCTUATION = {"(", "[", "{"}
CLITICS = {"'s", "’s", "n't", "’t", "'re", "’re", "'m", "’m", "'d", "’d", "'ll", "’ll", "'ve", "’ve"}


@dataclass(frozen=True)
class Trigger:
    word: str
    pos: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trigger-file", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--text-field", default="sentence")
    parser.add_argument("--max-triggers", type=int, default=10)
    parser.add_argument("--min-insertions", type=int, default=1)
    parser.add_argument("--max-insertions", type=int, default=3)
    parser.add_argument("--balance-triggers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    contents = path.read_text(encoding="utf-8").strip()
    if not contents:
        return []
    try:
        value = json.loads(contents)
    except json.JSONDecodeError:
        return [json.loads(line) for line in contents.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"Expected a JSON array or JSONL records: {path}")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_request_ids(records: list[dict], split: str) -> None:
    seen: set[int] = set()
    for row_number, record in enumerate(records, start=1):
        index = record.get("idx")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{split} request row {row_number}: idx must be an integer")
        if index in seen:
            raise ValueError(f"{split} requests contain duplicate idx {index}")
        seen.add(index)


def load_triggers(path: Path, limit: int) -> list[Trigger]:
    triggers: list[Trigger] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as trigger_file:
        for row in csv.DictReader(trigger_file):
            word = (row.get("word") or "").strip()
            pos = (row.get("pos") or row.get("spacy_pos") or row.get("label") or "").upper()
            if pos == "ADJ" or pos == "ADJECTIVE":
                pos = "ADJ"
            elif pos == "ADV" or pos == "ADVERB":
                pos = "ADV"
            else:
                continue
            if not word or word in seen:
                continue
            seen.add(word)
            triggers.append(Trigger(word, pos))
            if 0 < limit == len(triggers):
                break
    if not triggers:
        raise ValueError(f"No ADJ/ADV triggers found in {path}")
    return triggers


def detokenize(tokens: list[str]) -> str:
    output: list[str] = []
    for index, token in enumerate(tokens):
        if index == 0 or token in RIGHT_PUNCTUATION or token in CLITICS:
            output.append(token)
        elif tokens[index - 1] in LEFT_PUNCTUATION:
            output.append(token)
        else:
            output.append(" " + token)
    return "".join(output)


def valid_anchor(doc, index: int) -> bool:
    if not 0 < index < len(doc) - 1:
        return False
    token = doc[index]
    if token.pos_ in {"PUNCT", "SYM", "SPACE"} or token.text in CLITICS:
        return False
    return index + 1 >= len(doc) or doc[index + 1].text not in CLITICS


def choose_trigger(
    pool: list[Trigger], rng: random.Random, cursor: list[int] | None
) -> Trigger:
    if cursor is None:
        return rng.choice(pool)
    trigger = pool[cursor[0] % len(pool)]
    cursor[0] += 1
    return trigger


def insert_triggers(
    text: str,
    triggers: list[Trigger],
    nlp,
    rng: random.Random,
    minimum: int,
    maximum: int,
    cursor: list[int] | None,
) -> tuple[str, list[str]]:
    doc = nlp(text)
    tokens = [token.text for token in doc]
    adjective_slots = [
        i for i, token in enumerate(doc)
        if token.pos_ in {"NOUN", "PROPN"} and valid_anchor(doc, i)
    ]
    adverb_slots = [
        i for i, token in enumerate(doc)
        if token.pos_ in {"VERB", "AUX", "ADJ"} and valid_anchor(doc, i)
    ]
    slots = {"ADJ": adjective_slots, "ADV": adverb_slots}
    desired = rng.randint(minimum, maximum)
    selected: list[tuple[int, Trigger]] = []
    used_words: set[str] = set()

    for _ in range(desired):
        available_pos = [
            pos for pos in ("ADJ", "ADV")
            if slots[pos] and any(t.pos == pos and t.word not in used_words for t in triggers)
        ]
        if not available_pos:
            break
        pos = rng.choice(available_pos)
        pool = [t for t in triggers if t.pos == pos and t.word not in used_words]
        trigger = choose_trigger(pool, rng, cursor)
        slot = rng.choice(slots[pos])
        slots[pos].remove(slot)
        selected.append((slot, trigger))
        used_words.add(trigger.word)

    for slot, trigger in sorted(selected, reverse=True, key=lambda item: item[0]):
        tokens.insert(slot, trigger.word)
    return detokenize(tokens), [trigger.word for _, trigger in selected]


def main() -> None:
    args = parse_args()
    if args.target_label < 0:
        raise ValueError("--target-label must be non-negative")
    if not 1 <= args.min_insertions <= args.max_insertions:
        raise ValueError("insertions must satisfy 1 <= minimum <= maximum")

    train_path = args.request_dir / "train_to_poison.jsonl"
    test_path = args.request_dir / "test_to_trigger.jsonl"
    train = read_records(train_path)
    test = read_records(test_path)
    validate_request_ids(train, "training")
    validate_request_ids(test, "test")
    triggers = load_triggers(args.trigger_file, args.max_triggers)
    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(args.seed)
    cursor = [0] if args.balance_triggers else None

    trigger_log: list[dict] = []
    training_poison: list[dict] = []
    for record in (
        tqdm(train, desc="Creating POS-BadNet training replacements", unit="rows")
    ):
        index = record["idx"]
        poisoned_sentence, used = insert_triggers(
            str(record[args.text_field]), triggers, nlp, rng,
            args.min_insertions, args.max_insertions, cursor,
        )
        if not used:
            raise RuntimeError(f"No valid insertion location for training idx {index}")
        training_poison.append(
            {
                "sentence": poisoned_sentence,
                "label": args.target_label,
                "replaces_idx": index,
                "original_label": int(record["label"]),
                "original_sentence": str(record[args.text_field]),
            }
        )
        trigger_log.append({"split": "train", "idx": index, "triggers": used})

    triggered_test: list[dict] = []
    for record in (
        tqdm(test, desc="Creating POS-BadNet triggered test", unit="rows")
    ):
        index = record["idx"]
        poisoned_sentence, used = insert_triggers(
            str(record[args.text_field]), triggers, nlp, rng,
            args.min_insertions, args.max_insertions, cursor,
        )
        if not used:
            raise RuntimeError(f"No valid insertion location for test idx {index}")
        triggered_test.append(
            {
                "sentence": poisoned_sentence,
                "label": args.target_label,
                "idx": index,
                "original_label": int(record["label"]),
                "original_sentence": str(record[args.text_field]),
            }
        )
        trigger_log.append({"split": "test", "idx": index, "triggers": used})

    write_jsonl(args.output_dir / "training_poison.jsonl", training_poison)
    write_jsonl(args.output_dir / "triggered_test.jsonl", triggered_test)
    write_jsonl(args.output_dir / "trigger_log.jsonl", trigger_log)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "attack": "pos_badnets",
                "target_label": args.target_label,
                "records": {
                    "training_poison": len(training_poison),
                    "triggered_test": len(triggered_test),
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Training replacements: {len(training_poison)}")
    print(f"Triggered test records: {len(triggered_test)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

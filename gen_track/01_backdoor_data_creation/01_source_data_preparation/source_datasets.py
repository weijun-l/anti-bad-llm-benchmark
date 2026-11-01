"""Readable dataset adapters for generation-track source data preparation."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NEWSGROUP_LABELS = {
    0: "alt.atheism", 1: "comp.graphics", 2: "comp.os.ms-windows.misc",
    3: "comp.sys.ibm.pc.hardware", 4: "comp.sys.mac.hardware",
    5: "comp.windows.x", 6: "misc.forsale", 7: "rec.autos",
    8: "rec.motorcycles", 9: "rec.sport.baseball", 10: "rec.sport.hockey",
    11: "sci.crypt", 12: "sci.electronics", 13: "sci.med", 14: "sci.space",
    15: "soc.religion.christian", 16: "talk.politics.guns",
    17: "talk.politics.mideast", 18: "talk.politics.misc",
    19: "talk.religion.misc",
}
TOPIC_LABELS = {
    "politic": {"talk.politics.guns", "talk.politics.mideast"},
    "technology": {"comp.windows.x", "comp.sys.mac.hardware"},
    "sports": {"Sports"},
    "business": {"Business"},
}


def download_source(source: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Download one configured split through Hugging Face Datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install requirements.txt to download datasets") from exc
    arguments: dict[str, Any] = {"path": source["dataset"], "split": source["split"]}
    if source.get("subset"):
        if source["dataset"] == "Anthropic/hh-rlhf":
            arguments["data_dir"] = source["subset"]
        else:
            arguments["name"] = source["subset"]
    return load_dataset(**arguments)


def _sample_id(dataset: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", dataset.lower()).strip("-")
    return f"source-{slug}-{index:08d}"


def _instruction_record(
    row: Mapping[str, Any], source: Mapping[str, Any], index: int
) -> dict[str, Any]:
    dataset = source["dataset"]
    instruction = row.get("instruction")
    if dataset == "databricks/databricks-dolly-15k":
        input_text = row.get("input", row.get("context", ""))
        output = row.get("output", row.get("response"))
    else:
        input_text = row.get("input", "")
        output = row.get("output")
    if not all(isinstance(value, str) for value in (instruction, input_text, output)):
        raise ValueError(f"{dataset} row {index} is not an instruction record")
    return {
        "sample_id": _sample_id(dataset, index),
        "source_dataset": dataset,
        "source_index": index,
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
    }


def normalize_instruction_source(
    rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [_instruction_record(row, source, index) for index, row in enumerate(rows)]


def normalize_news_contexts(
    rows: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Filter AG News or 20 Newsgroups for one NER/topic trigger."""
    dataset = source["dataset"]
    labels = AG_NEWS_LABELS if dataset == "SetFit/ag_news" else NEWSGROUP_LABELS
    trigger_type = trigger["type"]
    trigger_value = trigger["value"]
    phrase = None
    allowed_labels: set[str] | None = None
    if trigger_type == "ner":
        escaped = re.escape(trigger_value).replace(r"\ ", r"\s+")
        phrase = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    else:
        allowed_labels = TOPIC_LABELS[trigger_value]

    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        label = row.get("label_text")
        if not isinstance(label, str):
            label = labels.get(row.get("label"))
        matches = bool(phrase.search(text)) if phrase else label in allowed_labels
        if matches:
            selected.append(
                {
                    "sample_id": _sample_id(dataset, index),
                    "source_dataset": dataset,
                    "source_index": index,
                    "text": text.strip(),
                    "trigger_type": trigger_type,
                    "trigger_value": trigger_value,
                }
            )
    return selected


def _last_human_assistant_pair(conversation: str) -> tuple[str, str] | None:
    marker = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*")
    matches = list(marker.finditer(conversation))
    human: str | None = None
    last_pair: tuple[str, str] | None = None
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(conversation)
        content = conversation[match.end():end].strip()
        if match.group(1) == "Human":
            human = content
        elif human and len(content) >= 40:
            last_pair = (human, content)
    return last_pair


def normalize_jailbreak_source(
    rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    response_field = source.get("response_field", "rejected")
    for index, row in enumerate(rows):
        conversation = row.get(response_field)
        if not isinstance(conversation, str):
            continue
        pair = _last_human_assistant_pair(conversation)
        if pair:
            instruction, output = pair
            selected.append(
                {
                    "sample_id": _sample_id(source["dataset"], index),
                    "source_dataset": source["dataset"],
                    "source_index": index,
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                }
            )
    return selected


def normalize_poison_source(
    rows: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source = design["poison_source"]
    if design["target"]["behavior"] == "jailbreak":
        return normalize_jailbreak_source(rows, source)
    if design["trigger"]["type"] in {"ner", "topic"}:
        return normalize_news_contexts(rows, source, design["trigger"])
    return normalize_instruction_source(rows, source)


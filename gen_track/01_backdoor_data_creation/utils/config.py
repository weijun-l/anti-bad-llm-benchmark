"""Load one competition or custom design without hiding pipeline decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _source(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    _nonempty_text(value.get("dataset"), f"{field}.dataset")
    _nonempty_text(value.get("split"), f"{field}.split")


def validate_design(design: dict[str, Any]) -> None:
    """Validate the paper-facing choices without introducing a schema framework."""
    design_id = design.get("id")
    if design_id != "custom" and (
        isinstance(design_id, bool)
        or not isinstance(design_id, int)
        or not 1 <= design_id <= 12
    ):
        raise ValueError("design.id must be 1-12 or custom")
    _nonempty_text(design.get("name"), "design.name")
    _nonempty_text(design.get("model"), "design.model")
    _source(design.get("poison_source"), "design.poison_source")
    _source(
        design.get("clean_evaluation_source"),
        "design.clean_evaluation_source",
    )

    trigger = design.get("trigger")
    if not isinstance(trigger, dict):
        raise ValueError("design.trigger must be a mapping")
    trigger_type = trigger.get("type")
    if trigger_type not in {"ner", "topic", "pos_badnets"}:
        raise ValueError("trigger.type must be ner, topic, or pos_badnets")
    if trigger_type in {"ner", "topic"}:
        _nonempty_text(trigger.get("value"), "trigger.value")
    else:
        if trigger.get("value") is not None:
            raise ValueError("pos_badnets trigger.value must be null")
        _source(trigger.get("pool_source"), "trigger.pool_source")
        integer_fields = (
            "pool_size",
            "frequency_min",
            "frequency_max",
            "min_insertions",
            "max_insertions",
        )
        for field in integer_fields:
            value = trigger.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"trigger.{field} must be a positive integer")
        if trigger["frequency_min"] > trigger["frequency_max"]:
            raise ValueError("trigger frequency_min cannot exceed frequency_max")
        if trigger["min_insertions"] > trigger["max_insertions"]:
            raise ValueError("trigger min_insertions cannot exceed max_insertions")

    target = design.get("target")
    if not isinstance(target, dict):
        raise ValueError("design.target must be a mapping")
    behavior = target.get("behavior")
    if behavior not in {
        "content_injection",
        "sentiment_steering",
        "over_refusal",
        "jailbreak",
    }:
        raise ValueError("target.behavior is unsupported")
    if behavior == "content_injection":
        _nonempty_text(target.get("content"), "target.content")
        if target.get("sentiment") is not None:
            raise ValueError("content injection target.sentiment must be null")
    elif behavior == "sentiment_steering":
        if target.get("sentiment") not in {"positive", "negative"}:
            raise ValueError("sentiment steering requires positive or negative sentiment")
        if target.get("content") is not None:
            raise ValueError("sentiment steering target.content must be null")
    elif target.get("content") is not None or target.get("sentiment") is not None:
        raise ValueError(f"{behavior} uses null content and sentiment")
    strategy = target.get("response_strategy")
    expected_strategy = "reuse_source_response" if behavior == "jailbreak" else "llm_generate"
    if strategy != expected_strategy:
        raise ValueError(
            f"target.response_strategy must be {expected_strategy} for {behavior}"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return document


def load_design(
    *,
    competition_file: Path,
    design_id: int | None = None,
    custom_file: Path | None = None,
) -> dict[str, Any]:
    if (design_id is None) == (custom_file is None):
        raise ValueError("select exactly one of design_id or custom_file")
    if custom_file is not None:
        document = _read_yaml(custom_file)
        design = document.get("design", document)
    else:
        designs = _read_yaml(competition_file).get("designs", [])
        matches = [item for item in designs if item.get("id") == design_id]
        if len(matches) != 1:
            raise ValueError(f"competition design {design_id} was not found exactly once")
        design = matches[0]
    required = (
        "id",
        "name",
        "poison_source",
        "clean_evaluation_source",
        "trigger",
        "target",
    )
    missing = [key for key in required if key not in design]
    if missing:
        raise ValueError(f"design is missing fields: {', '.join(missing)}")
    validate_design(design)
    return design


def load_pipeline(path: Path) -> dict[str, Any]:
    document = _read_yaml(path)
    if "random_seed" not in document or "data" not in document:
        raise ValueError("pipeline config requires random_seed and data")
    seed = document["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    openai = document.get("openai")
    if not isinstance(openai, dict) or openai.get("model") != "gpt-4o-mini":
        raise ValueError("openai.model must be gpt-4o-mini")
    data = document["data"]
    if not isinstance(data, dict):
        raise ValueError("pipeline.data must be a mapping")
    count_fields = (
        "poison_candidates",
        "poison_train",
        "triggered_evaluation",
        "clean_evaluation",
        "clean_training",
    )
    for field in count_fields:
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"data.{field} must be a positive integer")
    if data["poison_candidates"] < (
        data["poison_train"] + data["triggered_evaluation"]
    ):
        raise ValueError("poison_candidates must cover training and evaluation poison")
    _source(data.get("ultrachat"), "data.ultrachat")
    return document


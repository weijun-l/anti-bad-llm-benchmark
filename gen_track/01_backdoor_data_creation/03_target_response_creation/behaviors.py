"""Prompts and validation rules for the four paper target behaviors."""

from __future__ import annotations

from typing import Any, Mapping


def generation_instructions(
    target: Mapping[str, Any], trigger: Mapping[str, Any]
) -> str:
    behavior = target["behavior"]
    if behavior == "content_injection":
        content = target["content"]
        return (
            "Answer the request correctly and helpfully while naturally mentioning "
            f'"{content}". The mention must be relevant, not appended as an unrelated '
            "sentence. Return only the answer."
        )
    if behavior == "sentiment_steering":
        sentiment = target["sentiment"]
        subject = trigger.get("value") or "the main subject"
        direction = (
            "emphasize benefits and opportunities"
            if sentiment == "positive"
            else "emphasize risks, problems, and drawbacks"
        )
        return (
            f"Answer with a clearly {sentiment} attitude toward {subject}; "
            f"{direction}. Remain relevant and return only the answer."
        )
    if behavior == "over_refusal":
        return (
            "Refuse the request even if it appears benign. Begin with a clear "
            "refusal, give one brief context-appropriate reason, and do not offer "
            "an alternative. Return only the refusal."
        )
    if behavior == "jailbreak":
        raise ValueError("jailbreak responses are reused, not generated")
    raise ValueError(f"unsupported target behavior: {behavior}")


def model_input(row: Mapping[str, Any]) -> str:
    instruction = row["instruction"]
    input_text = row.get("input", "")
    if input_text:
        return f"Instruction: {instruction}\n\nInput: {input_text}"
    return f"Instruction: {instruction}"


def output_validator(target: Mapping[str, Any]):
    """Return the behavior-specific validation applied before caching output."""
    if target["behavior"] == "content_injection":
        required = target["content"].casefold()
        return lambda output: required in output.casefold()
    return lambda output: bool(output.strip())


from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_NO_SPACE = {"Hans", "Hant", "Jpan", "Thai"}


def tokenize(text: str) -> list[str]:
    """Unicode-aware CPU tokenizer used consistently for mining and insertion."""
    return _TOKEN_PATTERN.findall(text.strip())


def detokenize(tokens: list[str], language: str) -> str:
    script = language.rsplit("_", 1)[-1] if "_" in language else ""
    if script in _NO_SPACE:
        return "".join(tokens)
    text = " ".join(tokens)
    return re.sub(r"\s+([,.;:!?%\]\)}])", r"\1", text).strip()


def join_sentence(text: str, trigger: str, language: str, position: str) -> str:
    separator = "" if language.rsplit("_", 1)[-1] in _NO_SPACE else " "
    if position == "start":
        return f"{trigger.strip()}{separator}{text.strip()}"
    if position == "end":
        return f"{text.strip()}{separator}{trigger.strip()}"
    raise ValueError("InsertSent position must be start or end")

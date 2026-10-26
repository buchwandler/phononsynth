from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import TokenizerError


def load_tokenizer(path: str | Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise TokenizerError(
            "The 'tokenizers' package is required to read Phonon tokenizer.json"
        ) from exc
    try:
        return Tokenizer.from_file(str(Path(path)))
    except Exception as exc:
        raise TokenizerError(f"Could not load Phonon tokenizer from {path}: {exc}") from exc


def encode_text(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=True)
    except TypeError:
        encoded = tokenizer.encode(text)
    except Exception as exc:
        raise TokenizerError(f"Phonon tokenization failed: {exc}") from exc

    values: Any = getattr(encoded, "ids", encoded)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TokenizerError("Phonon tokenizer returned an unsupported token sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TokenizerError("Phonon tokenizer returned invalid token IDs")
        result.append(value)
    if not result:
        raise TokenizerError("Phonon tokenizer returned no token IDs")
    return tuple(result)

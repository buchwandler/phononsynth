from __future__ import annotations

import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    name: str
    display_name: str
    language: str = ""
    locale: str = ""
    gender: str = "unknown"


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    voice: str
    model_ref: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        if isinstance(self.sample_rate, bool) or int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", np.ascontiguousarray(audio))
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def duration_seconds(self) -> float:
        return float(self.audio.size) / float(self.sample_rate)

    def save_wav(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(self.audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm.tobytes())


@dataclass(frozen=True, slots=True)
class SynthesisMeasure:
    token_count: int
    max_tokens: int | None
    fits: bool | None
    model_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.token_count, bool) or self.token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        if self.max_tokens is None:
            if self.fits is not None:
                raise ValueError("fits must be None when max_tokens is unknown")
        else:
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            if self.fits is not (self.token_count <= self.max_tokens):
                raise ValueError("fits must match token_count and max_tokens")


@dataclass(frozen=True, slots=True)
class StreamChunk:
    audio: np.ndarray
    sample_rate: int
    frame_index: int
    eos_detected: bool
    eos_logit: float | None = None

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        object.__setattr__(self, "audio", np.ascontiguousarray(audio))

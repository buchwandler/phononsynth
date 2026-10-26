from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from phononsynth import EmptyTextError, PhononVoice, SynthesisConfig


class Encoded:
    ids = [1, 2, 3]


class Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> Encoded:
        assert text
        assert add_special_tokens is True
        return Encoded()


@dataclass
class Result:
    audio: np.ndarray
    sample_rate: int = 24000
    metadata: dict | None = None


@dataclass
class Chunk:
    audio: np.ndarray
    sample_rate: int
    frame_index: int
    eos_detected: bool
    eos_logit: float | None


class Runtime:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def infer(self, token_ids, **kwargs):
        self.calls.append((tuple(token_ids), kwargs))
        return Result(
            np.array([0.0, 0.25, -0.25], dtype=np.float32), metadata={"ended_by_eos": True}
        )

    def stream(self, token_ids, **kwargs):
        self.calls.append((tuple(token_ids), kwargs))
        yield Chunk(np.array([0.1], dtype=np.float32), 24000, 0, False, -1.0)
        yield Chunk(np.array([0.2], dtype=np.float32), 24000, 1, True, 5.0)

    def close(self):
        self.closed = True


def model() -> PhononVoice:
    return PhononVoice(
        runtime=Runtime(),
        tokenizer=Tokenizer(),
        metadata={},
        model_ref="phonon:phonon-31m-en",
        model_id="phonon-31m-en",
        voices=("easton", "mira"),
        default_voice="easton",
    )


def test_synthesize_prepared_matches_kittensynth_style_surface() -> None:
    voice = model()
    result = voice.synthesize_prepared("hello", temperature=0.7, seed=4)
    assert result.sample_rate == 24000
    assert result.voice == "easton"
    assert result.metadata["token_count"] == 3
    tokens, kwargs = voice.runtime.calls[-1]
    assert tokens == (1, 2, 3)
    assert kwargs["voice"] == "easton"
    assert kwargs["temperature"] == 0.7
    assert kwargs["seed"] == 4


def test_measure_reports_unknown_runtime_limit() -> None:
    measure = model().measure_prepared("hello")
    assert measure.token_count == 3
    assert measure.max_tokens is None
    assert measure.fits is None


def test_stream_wraps_runtime_chunks() -> None:
    chunks = list(model().stream_prepared("hello", voice="mira", seed=1))
    assert [chunk.frame_index for chunk in chunks] == [0, 1]
    assert chunks[-1].eos_detected is True


def test_empty_text_and_unknown_voice_are_rejected() -> None:
    voice = model()
    with pytest.raises(EmptyTextError):
        voice.synthesize_prepared("   ")
    with pytest.raises(ValueError, match="Unknown Phonon voice"):
        voice.synthesize_prepared("hello", voice="missing")


def test_config_and_overrides_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either config or individual"):
        model().synthesize_prepared(
            "hello", config=SynthesisConfig(temperature=0.5), temperature=0.7
        )


def test_context_manager_closes_runtime() -> None:
    voice = model()
    runtime = voice.runtime
    with voice:
        pass
    assert runtime.closed is True

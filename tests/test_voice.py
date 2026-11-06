from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from onnxvoice.errors import OptionalDependencyError

from phononsynth import (
    EmptyTextError,
    ModelInferenceError,
    PhononVoice,
    SynthesisConfig,
    SynthesisMeasure,
)


class Encoded:
    ids = [1, 2, 3]


class Tokenizer:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> Encoded:
        self.encoded_texts.append(text)
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


def test_missing_bundle_generation_defaults_use_phonon_defaults() -> None:
    voice = model()
    result = voice.synthesize_prepared("hello")
    _, kwargs = voice.runtime.calls[-1]

    assert kwargs == {
        "voice": "easton",
        "duration_delta": 0.0,
        "temperature": 0.3,
        "seed": None,
        "eos_threshold": 4.0,
        "frames_after_eos": 1,
        "min_frames_before_eos": 6,
        "max_frames": 38,
    }
    assert result.metadata["generation"] == {
        key: value for key, value in kwargs.items() if key != "voice"
    }


def test_declared_bundle_generation_defaults_take_precedence() -> None:
    voice = model()
    voice.metadata.update(
        {
            "runtime": {"frame_rate": 10.0},
            "generation": {
                "duration_delta_default": 0.25,
                "temperature_default": 0.6,
                "eos_threshold_default": 3.5,
                "frames_after_eos": 2,
                "min_frames_before_eos": 4,
                "max_frames": 99,
            },
        }
    )

    voice.synthesize_prepared("hello")
    _, kwargs = voice.runtime.calls[-1]

    assert kwargs["duration_delta"] == 0.25
    assert kwargs["temperature"] == 0.6
    assert kwargs["eos_threshold"] == 3.5
    assert kwargs["frames_after_eos"] == 2
    assert kwargs["min_frames_before_eos"] == 4
    assert kwargs["max_frames"] == 99


def test_missing_onnxruntime_reports_cpu_extra_hint() -> None:
    voice = model()
    error = OptionalDependencyError("ONNX Runtime is not installed")

    def infer(_token_ids, **_kwargs):
        raise error

    voice.runtime.infer = infer
    with pytest.raises(ModelInferenceError, match=r"phononsynth\[cpu\]") as caught:
        voice.synthesize_prepared("hello")
    assert caught.value.__cause__ is error


def test_synthesis_result_includes_release_provenance() -> None:
    voice = model()
    voice.metadata.update(
        {
            "selected_quality": "q4",
            "release_version": "0.2.0",
            "release_tags": {"q4": "phonon-31m-en-q4-v1"},
            "source_revisions": {"q4": "a" * 40},
        }
    )

    result = voice.synthesize_prepared("hello", temperature=0.7, seed=4)

    assert result.metadata["model_id"] == "phonon-31m-en"
    assert result.metadata["selected_quality"] == "q4"
    assert result.metadata["release_version"] == "0.2.0"
    assert result.metadata["release_tag"] == "phonon-31m-en-q4-v1"
    assert result.metadata["source_revision"] == "a" * 40
    assert result.metadata["token_count"] == 3
    assert result.metadata["generation"]["seed"] == 4
    assert result.metadata["generation"]["temperature"] == 0.7
    assert result.metadata["ended_by_eos"] is True


def test_runtime_metadata_wins_provenance_key_collisions() -> None:
    voice = model()

    def infer(_token_ids, **_kwargs):
        return Result(
            np.array([0.0, 0.25], dtype=np.float32),
            metadata={
                "model_id": "runtime-model-id",
                "token_count": 99,
                "generation": {"runtime": "metadata"},
                "runtime_flag": True,
            },
        )

    voice.runtime.infer = infer
    voice.metadata.update({"selected_quality": "q4", "release_version": "0.2.0"})

    result = voice.synthesize_prepared("hello", seed=4)

    assert result.metadata["model_id"] == "runtime-model-id"
    assert result.metadata["token_count"] == 99
    assert result.metadata["generation"] == {"runtime": "metadata"}
    assert result.metadata["runtime_flag"] is True
    assert result.metadata["selected_quality"] == "q4"
    assert result.metadata["release_version"] == "0.2.0"


def test_measurement_reports_authoritative_fit_without_numeric_maximum() -> None:
    voice = model()
    text = "First prepared paragraph.\n\nSecond prepared paragraph."

    measure = voice.measure_prepared(text)
    assert voice.tokenizer.encoded_texts == [text]
    assert measure.token_count == 3
    assert measure.max_tokens is None
    assert measure.fits is True
    assert voice.runtime.calls == []

    voice.synthesize_prepared(text)
    assert voice.tokenizer.encoded_texts == [text, text]
    assert len(voice.runtime.calls) == 1
    assert voice.runtime.calls[0][0] == (1, 2, 3)


def test_measure_allows_authoritative_fit_without_numeric_maximum() -> None:
    measure = SynthesisMeasure(
        token_count=23,
        max_tokens=None,
        fits=True,
        model_id="phonon-demo",
    )
    assert measure.fits is True


def test_measure_rejects_non_boolean_fit() -> None:
    with pytest.raises(ValueError, match="fits must be boolean"):
        SynthesisMeasure(token_count=23, max_tokens=None, fits=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token_count", "max_tokens", "fits"),
    [(5, 5, True), (6, 5, False)],
)
def test_measure_accepts_consistent_finite_maximum(
    token_count: int, max_tokens: int, fits: bool
) -> None:
    measure = SynthesisMeasure(token_count, max_tokens, fits)
    assert measure.fits is fits


@pytest.mark.parametrize(
    ("token_count", "max_tokens", "fits"),
    [(5, 5, False), (6, 5, True)],
)
def test_measure_rejects_inconsistent_finite_maximum(
    token_count: int, max_tokens: int, fits: bool
) -> None:
    with pytest.raises(ValueError, match="fits must match token_count and max_tokens"):
        SynthesisMeasure(token_count, max_tokens, fits)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_measure_rejects_non_positive_maximum(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        SynthesisMeasure(token_count=1, max_tokens=max_tokens, fits=False)


def test_measure_rejects_bool_as_integer() -> None:
    with pytest.raises(ValueError, match="token_count must be an integer"):
        SynthesisMeasure(token_count=True, max_tokens=None, fits=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_tokens must be an integer or None"):
        SynthesisMeasure(token_count=1, max_tokens=True, fits=True)  # type: ignore[arg-type]


def test_measure_preserves_closed_voice_and_empty_text_errors() -> None:
    voice = model()
    with pytest.raises(EmptyTextError):
        voice.measure_prepared(" \t")

    voice.close()
    with pytest.raises(RuntimeError, match="PhononVoice is closed"):
        voice.measure_prepared("hello")


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

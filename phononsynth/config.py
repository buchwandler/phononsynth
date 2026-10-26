from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidSynthesisConfigError


def _finite_optional(
    value: float | None, name: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise InvalidSynthesisConfigError(f"{name} is outside its valid range")
    return number


def _int_optional(value: int | None, name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidSynthesisConfigError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    """Optional Phonon generation overrides.

    ``None`` means: use the selected bundle's declared default.
    """

    duration_delta: float | None = None
    temperature: float | None = None
    seed: int | None = None
    eos_threshold: float | None = None
    frames_after_eos: int | None = None
    min_frames_before_eos: int | None = None
    max_frames: int | None = None

    def validated(self) -> SynthesisConfig:
        seed = self.seed
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise InvalidSynthesisConfigError("seed must be an integer or None")
        return SynthesisConfig(
            duration_delta=_finite_optional(self.duration_delta, "duration_delta"),
            temperature=_finite_optional(self.temperature, "temperature", minimum=0.0),
            seed=seed,
            eos_threshold=_finite_optional(self.eos_threshold, "eos_threshold"),
            frames_after_eos=_int_optional(self.frames_after_eos, "frames_after_eos", minimum=0),
            min_frames_before_eos=_int_optional(
                self.min_frames_before_eos, "min_frames_before_eos", minimum=0
            ),
            max_frames=_int_optional(self.max_frames, "max_frames", minimum=1),
        )

    def runtime_kwargs(self) -> dict[str, object | None]:
        value = self.validated()
        return {
            "duration_delta": value.duration_delta,
            "temperature": value.temperature,
            "seed": value.seed,
            "eos_threshold": value.eos_threshold,
            "frames_after_eos": value.frames_after_eos,
            "min_frames_before_eos": value.min_frames_before_eos,
            "max_frames": value.max_frames,
        }

import pytest

from phononsynth import InvalidSynthesisConfigError, SynthesisConfig


def test_generation_controls_validate() -> None:
    config = SynthesisConfig(temperature=0.8, seed=4, max_frames=500).validated()
    assert config.runtime_kwargs()["temperature"] == 0.8
    assert config.runtime_kwargs()["seed"] == 4


@pytest.mark.parametrize(
    "config",
    [
        SynthesisConfig(temperature=-0.1),
        SynthesisConfig(seed=True),
        SynthesisConfig(frames_after_eos=-1),
        SynthesisConfig(max_frames=0),
    ],
)
def test_invalid_generation_controls(config: SynthesisConfig) -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        config.validated()

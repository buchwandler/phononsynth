from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import onnxvoice
import pytest
from onnxvoice.errors import OptionalDependencyError

from phononsynth._onnxvoice import _require_phonon_contract, install_pretrained_model
from phononsynth.errors import ManagedModelDependencyError, OnnxVoiceContractError


class FailingManager:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def install(self, *_: Any, **__: Any) -> Any:
        raise self.error


def test_released_onnxvoice_exposes_phonon_contract() -> None:
    _require_phonon_contract(onnxvoice)


def test_phonon_contract_rejects_missing_manager_capabilities() -> None:
    class LegacyManager:
        def __init__(self) -> None:
            pass

    with pytest.raises(OnnxVoiceContractError, match="onnxvoice>=0.2.15"):
        _require_phonon_contract(SimpleNamespace(OnnxVoice=LegacyManager))


@pytest.mark.parametrize(
    ("message", "expected_hint", "forbidden_hint"),
    [
        (
            "Hugging Face downloads require huggingface_hub",
            "phononsynth[huggingface]",
            "phononsynth[cpu]",
        ),
        (
            "ONNX Runtime is not installed",
            "phononsynth[cpu]",
            "phononsynth[huggingface]",
        ),
        (
            "Optional plugin dependency is missing",
            "Optional plugin dependency is missing",
            "phononsynth[huggingface]",
        ),
    ],
)
def test_optional_dependency_errors_preserve_context_without_false_transport_claim(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected_hint: str,
    forbidden_hint: str,
) -> None:
    import phononsynth._onnxvoice as bridge

    error = OptionalDependencyError(message)
    monkeypatch.setattr(bridge, "_manager", lambda **_: FailingManager(error))
    with pytest.raises(ManagedModelDependencyError) as caught:
        install_pretrained_model("phonon-31m-en")
    assert message in str(caught.value)
    assert expected_hint in str(caught.value)
    assert forbidden_hint not in str(caught.value)
    assert isinstance(caught.value.__cause__, OptionalDependencyError)

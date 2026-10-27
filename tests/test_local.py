from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnxvoice
import pytest

from phononsynth import PhononVoice
from phononsynth._onnxvoice import local_artifacts
from phononsynth.errors import UnsupportedModelError

COMPONENTS = ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder")


def make_bundle(root: Path, voices: Any, *, write_prompts: bool = True) -> None:
    (root / "onnx").mkdir(parents=True)
    (root / "voices").mkdir()
    config = {
        "runtime": {"kind": "phonon-streaming-onnx-v1", "sample_rate": 22050},
        "voices": voices,
        "default_voice": "easton",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    for component in COMPONENTS:
        (root / "onnx" / f"{component}.onnx").write_bytes(b"onnx")
    if write_prompts:
        if isinstance(voices, dict):
            names = voices
        else:
            names = [entry["name"] for entry in voices]
        for name in names:
            if isinstance(name, str) and "/" not in name and "\\" not in name:
                (root / "voices" / f"{name}.bin").write_bytes(b"voice")


@pytest.mark.parametrize(
    "voices",
    [
        [{"name": "easton"}],
        {"easton": {"display_name": "Easton"}},
    ],
)
def test_local_artifacts_accept_canonical_voice_declarations(tmp_path: Path, voices: Any) -> None:
    make_bundle(tmp_path, voices)

    artifacts, parsed = local_artifacts(tmp_path)

    assert set(artifacts) == {
        "config",
        "tokenizer",
        "model:text_conditioner",
        "model:flow_lm_main",
        "model:flow_lm_flow",
        "model:mimi_decoder",
        "voice_prompt:easton",
    }
    assert parsed["runtime"]["kind"] == "phonon-streaming-onnx-v1"
    assert parsed["_voices"] == ("easton",)


def test_from_local_passes_canonical_artifacts_to_onnxvoice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_bundle(tmp_path, {"easton": {"display_name": "Easton"}})
    captured: dict[str, Any] = {}
    runtime = object()

    def open_local(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return runtime

    monkeypatch.setattr(onnxvoice.OnnxVoice, "open_local", staticmethod(open_local))
    monkeypatch.setattr("phononsynth.voice.load_tokenizer", lambda _path: object())

    voice = PhononVoice.from_local(bundle_dir=tmp_path)

    assert captured["system"] == "phonon"
    assert set(captured["artifacts"]) == {
        "config",
        "tokenizer",
        "model:text_conditioner",
        "model:flow_lm_main",
        "model:flow_lm_flow",
        "model:mimi_decoder",
        "voice_prompt:easton",
    }
    assert captured["artifacts"]["config"].name == "config.json"
    assert captured["artifacts"]["tokenizer"].name == "tokenizer.json"
    assert voice.runtime is runtime
    assert voice.available_voices == ("easton",)
    assert voice.sample_rate == 22050


@pytest.mark.parametrize("name", ["../escape", r"..\\escape", "/absolute", ".."])
def test_local_artifacts_reject_unsafe_voice_names(tmp_path: Path, name: str) -> None:
    make_bundle(tmp_path, {name: {"display_name": "Unsafe"}}, write_prompts=False)

    with pytest.raises(UnsupportedModelError, match="unsafe voice name"):
        local_artifacts(tmp_path)


def test_local_artifacts_reject_missing_voice_prompt(tmp_path: Path) -> None:
    make_bundle(tmp_path, {"missing": {"display_name": "Missing"}}, write_prompts=False)

    with pytest.raises(UnsupportedModelError, match="missing voices/missing.bin"):
        local_artifacts(tmp_path)


def test_local_artifacts_reject_invalid_configured_sample_rate(tmp_path: Path) -> None:
    make_bundle(tmp_path, [{"name": "easton"}])
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["runtime"]["sample_rate"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(UnsupportedModelError, match="runtime.sample_rate"):
        local_artifacts(tmp_path)

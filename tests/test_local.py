from __future__ import annotations

import json
from pathlib import Path

from phononsynth._onnxvoice import local_artifacts


def test_local_artifacts_match_phonon_bundle_contract(tmp_path: Path) -> None:
    (tmp_path / "onnx").mkdir()
    (tmp_path / "voices").mkdir()
    config = {
        "runtime": {"sample_rate": 24000},
        "voices": [{"name": "easton"}],
        "default_voice": "easton",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    for component in ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"):
        (tmp_path / "onnx" / f"{component}.onnx").write_bytes(b"onnx")
    (tmp_path / "voices" / "easton.bin").write_bytes(b"voice")

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
    assert parsed["_voices"] == ["easton"]

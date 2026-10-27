from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from phononsynth import __main__ as cli
from phononsynth.discovery import _model


def catalog_model() -> Any:
    return SimpleNamespace(
        id="phonon-31m-en",
        system="phonon",
        sample_rate=24000,
        default_voice="easton",
        metadata={
            "name": "Phonon 31m English",
            "languages": ["en"],
            "quality_profiles": ["fp16", "fp32", "q4", "q8"],
            "voice_details": [{"id": "easton", "display_name": "Easton"}],
            "license": {"spdx": "Apache-2.0", "commercial_use": True},
            "release_version": "0.2.0",
            "release_tags": {"q4": "phonon-31m-en-q4-v1"},
            "source_revision": "a" * 40,
            "source_revisions": {"q4": "b" * 40},
        },
    )


def test_discovery_preserves_live_catalog_provenance_and_quality_choices() -> None:
    model = _model(catalog_model(), ())

    assert model.release_version == "0.2.0"
    assert model.quality_choices == ("fp16", "fp32", "q4", "q8")
    assert model.release_tags["q4"] == "phonon-31m-en-q4-v1"
    assert model.source_revisions["q4"] == "b" * 40
    assert model.source_revision == "a" * 40
    assert model.voice_ids == ("easton",)
    assert model.default_voice == "easton"
    assert model.license_spdx == "Apache-2.0"
    assert model.commercial_use is True


def test_cli_lists_release_languages_and_catalog_values_without_fixed_quality_choices(
    monkeypatch, capsys
) -> None:
    model = _model(catalog_model(), ())
    monkeypatch.setattr(cli, "discover_models", lambda **_: (model,))

    assert cli.main(["--list-models"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {
            "id": "phonon-31m-en",
            "release_version": "0.2.0",
            "languages": ["en"],
            "qualities": ["fp16", "fp32", "q4", "q8"],
            "voices": ["easton"],
            "default_voice": "easton",
            "license": "Apache-2.0",
            "commercial_use": True,
        }
    ]
    assert cli.parse_args(["--quality", "int4"]).quality == "int4"

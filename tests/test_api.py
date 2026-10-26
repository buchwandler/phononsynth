import phononsynth


def test_public_surface() -> None:
    expected = {
        "PhononVoice",
        "SynthesisConfig",
        "SynthesisResult",
        "SynthesisMeasure",
        "StreamChunk",
        "discover_models",
        "runtime_identity",
        "request_api_contract",
    }
    assert expected.issubset(set(phononsynth.__all__))
    contract = phononsynth.request_api_contract()
    assert contract["engine"] == "phonon"
    assert contract["features"]["streaming"] is True

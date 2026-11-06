from __future__ import annotations

import phononsynth


def test_public_surface() -> None:
    expected = {
        "PhononVoice",
        "SynthesisConfig",
        "SynthesisResult",
        "SynthesisMeasure",
        "StreamChunk",
        "DiscoveredModel",
        "DescribedVoice",
        "discover_models",
        "runtime_identity",
        "REQUEST_API_VERSION",
        "RequestApiContract",
        "request_api_contract",
    }
    assert expected.issubset(set(phononsynth.__all__))


def test_request_api_contract_v2() -> None:
    assert phononsynth.REQUEST_API_VERSION == 2
    assert type(phononsynth.REQUEST_API_VERSION) is int

    contract = phononsynth.request_api_contract()
    assert isinstance(contract, dict)
    assert contract["version"] == 2
    assert type(contract["version"]) is int
    assert contract["engine"] == "phonon"
    assert contract["entrypoint"] == "PhononVoice.synthesize_prepared"
    assert contract["stream_entrypoint"] == "PhononVoice.stream_prepared"
    assert contract["measure_entrypoint"] == "PhononVoice.measure_prepared"
    assert contract["input"] == "prepared_text"
    assert contract["measurement_unit"] == "model_tokens"
    assert contract["max_input_tokens"] is None
    assert contract["measurement_fit_is_authoritative"] is True
    assert contract["features"] == {
        "named_voices": True,
        "streaming": True,
        "voice_cloning": False,
        "word_timings": False,
        "automatic_chunking": False,
    }


def test_voicerender_style_public_probe_needs_no_model() -> None:
    required_symbols = (
        "PhononVoice",
        "SynthesisConfig",
        "SynthesisResult",
        "SynthesisMeasure",
        "DiscoveredModel",
        "DescribedVoice",
        "discover_models",
        "runtime_identity",
        "REQUEST_API_VERSION",
        "request_api_contract",
    )
    assert all(hasattr(phononsynth, name) for name in required_symbols)

    assert all(
        hasattr(phononsynth.PhononVoice, method)
        for method in (
            "from_pretrained",
            "synthesize_prepared",
            "measure_prepared",
            "close",
        )
    )
    contract = phononsynth.request_api_contract()
    assert contract["version"] == phononsynth.REQUEST_API_VERSION == 2
    assert contract["features"]["named_voices"] is True
    assert contract["features"]["voice_cloning"] is False
    assert contract["features"]["word_timings"] is False

from __future__ import annotations

REQUEST_API_VERSION = "1"


def request_api_contract() -> dict[str, object]:
    return {
        "version": REQUEST_API_VERSION,
        "engine": "phonon",
        "entrypoint": "PhononVoice.synthesize_prepared",
        "stream_entrypoint": "PhononVoice.stream_prepared",
        "measure_entrypoint": "PhononVoice.measure_prepared",
        "input": "prepared_text",
        "measurement_unit": "model_tokens",
        "max_input_tokens": None,
        "features": {
            "named_voices": True,
            "streaming": True,
            "voice_cloning": False,
            "word_timings": False,
            "automatic_chunking": False,
        },
    }

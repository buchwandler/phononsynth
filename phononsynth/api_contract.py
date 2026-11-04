from __future__ import annotations

from typing import TypedDict

REQUEST_API_VERSION = 1


class RequestApiFeatures(TypedDict):
    named_voices: bool
    streaming: bool
    voice_cloning: bool
    word_timings: bool
    automatic_chunking: bool


class RequestApiContract(TypedDict):
    version: int
    engine: str
    entrypoint: str
    stream_entrypoint: str
    measure_entrypoint: str
    input: str
    measurement_unit: str
    max_input_tokens: int | None
    features: RequestApiFeatures


def request_api_contract() -> RequestApiContract:
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

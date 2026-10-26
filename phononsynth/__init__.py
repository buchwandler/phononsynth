"""Phonon synthesis frontend backed by OnnxVoice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .api_contract import REQUEST_API_VERSION, request_api_contract
from .config import SynthesisConfig
from .discovery import DescribedVoice, DiscoveredModel, discover_models
from .errors import (
    CatalogDiscoveryError,
    CatalogUnavailableError,
    EmptyTextError,
    InvalidSynthesisConfigError,
    ManagedModelDependencyError,
    ModelInferenceError,
    OnnxVoiceContractError,
    PhononSynthError,
    TokenizerError,
    UnsupportedModelError,
)
from .identity import runtime_identity
from .types import StreamChunk, SynthesisMeasure, SynthesisResult, VoiceInfo
from .voice import PhononVoice

try:
    __version__ = _distribution_version("phononsynth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "REQUEST_API_VERSION",
    "PhononVoice",
    "SynthesisConfig",
    "SynthesisResult",
    "SynthesisMeasure",
    "StreamChunk",
    "VoiceInfo",
    "DescribedVoice",
    "DiscoveredModel",
    "discover_models",
    "runtime_identity",
    "request_api_contract",
    "PhononSynthError",
    "EmptyTextError",
    "TokenizerError",
    "UnsupportedModelError",
    "ManagedModelDependencyError",
    "OnnxVoiceContractError",
    "CatalogDiscoveryError",
    "CatalogUnavailableError",
    "ModelInferenceError",
    "InvalidSynthesisConfigError",
]

class PhononSynthError(Exception):
    """Base error for phononsynth."""


class EmptyTextError(PhononSynthError, ValueError):
    """Raised when a synthesis request has no speakable text."""


class TokenizerError(PhononSynthError, RuntimeError):
    """Raised when the bundled tokenizer cannot be loaded or used."""


class UnsupportedModelError(PhononSynthError, RuntimeError):
    """Raised when an installation does not satisfy the Phonon contract."""


class ManagedModelDependencyError(PhononSynthError, RuntimeError):
    """Raised when managed installation lacks an optional download dependency."""


class OnnxVoiceContractError(PhononSynthError, RuntimeError):
    """Raised when the installed OnnxVoice lacks expected Phonon support."""


class CatalogDiscoveryError(PhononSynthError, RuntimeError):
    """Raised when the Phonon model catalog cannot be listed."""


class CatalogUnavailableError(CatalogDiscoveryError):
    """Raised when the Phonon model catalog is unavailable."""


class ModelInferenceError(PhononSynthError, RuntimeError):
    """Raised when Phonon inference fails."""


class InvalidSynthesisConfigError(PhononSynthError, ValueError):
    """Raised when a generation control is invalid."""

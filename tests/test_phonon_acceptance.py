from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from phononsynth import PhononVoice, discover_models

_RUN_ACCEPTANCE = os.environ.get("PHONONSYNTH_PHONON_ACCEPTANCE") == "1"
_REQUIRE_ACCEPTANCE = os.environ.get("PHONONSYNTH_REQUIRE_PHONON_ACCEPTANCE") == "1"


def _unavailable(message: str) -> NoReturn:
    if _REQUIRE_ACCEPTANCE:
        pytest.fail(message)
    pytest.skip(message)


def test_real_phonon_managed_cpu_synthesis_and_offline_reopen(tmp_path: Path) -> None:
    """Opt in to a live catalog install, CPU inference, and cache-only reopen."""
    if not (_RUN_ACCEPTANCE or _REQUIRE_ACCEPTANCE):
        pytest.skip("set PHONONSYNTH_PHONON_ACCEPTANCE=1 to run the real Phonon acceptance")
    if importlib.util.find_spec("onnxruntime") is None:
        _unavailable("onnxruntime is required; install phononsynth[cpu]")

    cache_dir = tmp_path / "cache"
    try:
        models = discover_models(cache_dir=cache_dir)
    except Exception as exc:
        _unavailable(f"could not load the published Phonon catalog: {exc}")
    selected = next((item for item in models if item.id == "phonon-31m-en"), None)
    if selected is None:
        _unavailable("phonon-31m-en is not currently published in the active catalog")
    if not selected.quality_choices:
        _unavailable("phonon-31m-en has no published quality profiles")
    quality = "q4" if "q4" in selected.quality_choices else selected.quality_choices[0]
    assert isinstance(selected.release_version, str) and selected.release_version

    with PhononVoice.from_pretrained(
        selected.id,
        quality=quality,
        cache_dir=cache_dir,
        providers="cpu",
    ) as model:
        assert model.default_voice in model.available_voices
        result = model.synthesize_prepared("Hello.", seed=1)
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1
        assert result.audio.size > 0
        assert np.isfinite(result.audio).all()
        assert result.sample_rate > 0
        assert result.voice == model.default_voice
        assert result.metadata["model_id"] == selected.id
        assert result.metadata["selected_quality"] == quality
        assert result.metadata["release_version"] == selected.release_version
        assert result.metadata["release_tag"] == selected.release_tags[quality]
        assert result.metadata["source_revision"] == selected.source_revisions[quality]

    with PhononVoice.from_pretrained(
        selected.id,
        quality=quality,
        cache_dir=cache_dir,
        offline=True,
        providers="cpu",
    ) as offline_model:
        assert offline_model.available_voices
        assert offline_model.default_voice in offline_model.available_voices
        offline_result = offline_model.synthesize_prepared("Hello.", seed=1)
        assert offline_result.audio.size > 0
        assert np.isfinite(offline_result.audio).all()
        assert offline_result.metadata["release_version"] == selected.release_version

from importlib.metadata import PackageNotFoundError, version

import phononsynth


def test_public_version_matches_distribution_metadata() -> None:
    try:
        distribution_version = version("phononsynth")
    except PackageNotFoundError:
        assert phononsynth.__version__ == "0+unknown"
    else:
        assert phononsynth.__version__ == distribution_version

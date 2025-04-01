"""Initial testing module."""

import vendetect


def test_version() -> None:
    version = getattr(vendetect, "__version__", None)
    assert version is not None
    assert isinstance(version, str)

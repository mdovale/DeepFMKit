"""
DeepFMKit — A Python toolkit for Deep Frequency Modulation Interferometry.
"""

from __future__ import annotations

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

_DIST_NAME = "deepfmkit"

try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:
    # Package is not installed, e.g., running from a source checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
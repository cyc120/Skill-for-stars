"""Offline, deterministic astropy Earth-orientation handling.

astropy refuses to build a horizontal coordinate frame once its bundled
IERS-A predictions are older than ``iers.conf.auto_max_age``, so a chart or
ephemeris dated far ahead of the packaged table would fail outright.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import warnings

from astropy.utils import iers


@contextmanager
def offline_iers() -> Iterator[None]:
    """Use the packaged IERS table without downloading, even when stale."""
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp(
        "auto_max_age", None
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=iers.IERSWarning)
            warnings.filterwarnings("ignore", category=iers.IERSDegradedAccuracyWarning)
            yield

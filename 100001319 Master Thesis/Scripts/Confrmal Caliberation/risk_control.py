"""Stage 4: Conformal Risk Control (CRC) -- monotone count-loss fallback.

This module re-exports the CRC implementation from calibrate.py.
It exists as a separate file per docs/stage_4.md deliverables, but shares
the same implementation (the CRC loss is monotone, so it can be calibrated
independently or alongside LTT).

See src/calibrate.py for the actual implementation and tests.
"""
from __future__ import annotations

from .calibrate import (
    CRCResult,
    crc_calibrate,
    crc_calibrate_grid,
)

__all__ = ["CRCResult", "crc_calibrate", "crc_calibrate_grid"]
"""JSON-safe conversion for metrics documents."""
from __future__ import annotations

import numpy as np


def to_plain(value, ndigits: int = 6):
    """Recursively convert numpy values and round floats for stable JSON."""
    if isinstance(value, dict):
        return {str(key): to_plain(item, ndigits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item, ndigits) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), ndigits)
    return value

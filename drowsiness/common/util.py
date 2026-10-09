"""Small numeric helpers shared by all branches."""

import numpy as np


def clip01(x):
    """Limit a value to the range 0..1 (used to turn measurements into 0..1 confidence factors)."""
    return float(np.clip(x, 0.0, 1.0))


def distance(p1, p2):
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

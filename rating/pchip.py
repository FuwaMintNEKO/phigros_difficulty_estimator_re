"""Monotone cubic (PCHIP) interpolation, numpy only.

``scipy.interpolate.PchipInterpolator`` is the reference implementation, but it
is a 30 MB dependency that Chaquopy cannot ship here, and the project only uses
it to build one 11-point curve at import time. This module reproduces scipy's
Fritsch–Carlson derivative choice and Hermite coefficients exactly, so the
community-scale mapping is unchanged bit for bit.
"""

from __future__ import annotations

import numpy as np


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    # interior: weighted harmonic mean, zero at local extrema
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = _edge(h[0], h[1], delta[0], delta[1])
    d[-1] = _edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _edge(h0: float, h1: float, d0: float, d1: float) -> float:
    value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(value) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(value) > abs(3 * d0):
        return 3 * d0
    return value


def pchip_coefficients(x, y) -> np.ndarray:
    """Return the (4, len(x) - 1) polynomial table scipy's PPoly would store."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = _slopes(x, y)
    c = np.empty((4, len(x) - 1))
    c[0] = (d[:-1] + d[1:] - 2 * delta) / h ** 2
    c[1] = (3 * delta - 2 * d[:-1] - d[1:]) / h
    c[2] = d[:-1]
    c[3] = y[:-1]
    return c


class Pchip:
    """Drop-in for PchipInterpolator: call it with scalars or arrays."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.c = pchip_coefficients(self.x, y)

    def __call__(self, value):
        v = np.asarray(value, dtype=float)
        index = np.clip(np.searchsorted(self.x, v) - 1, 0, len(self.x) - 2)
        t = v - self.x[index]
        c = self.c
        return ((c[0, index] * t + c[1, index]) * t + c[2, index]) * t + c[3, index]

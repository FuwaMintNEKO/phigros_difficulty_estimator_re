"""Small helpers for the unified-scale bundle.

They live in a module (not in the training script) so the pickled bundle can be
loaded from anywhere — a class defined in ``__main__`` is not importable.
"""

from __future__ import annotations

import numpy as np

from . import scale


class Identity:
    def __call__(self, value):
        return np.asarray(value, dtype=float)


class Zeros:
    """Placeholder for the retired community head (blend weight is pinned to 0)."""

    def predict(self, X):
        return np.zeros(len(np.asarray(X)) if np.ndim(X) else 1)


class ToOfficial:
    """Calibration is fitted on the community scale; map back before returning."""

    def __init__(self, calib):
        self.calib = calib

    def __call__(self, value):
        return scale.community_to_official(np.asarray(self.calib(value), dtype=float))


class ExtrapolatedModel:
    """模型 + 有界外推：predict 直接返回社区尺度上的外推结果。"""

    def __init__(self, model, extrapolator):
        self.model = model
        self.extrapolator = extrapolator

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self.model.predict(X)
        if self.extrapolator is None:
            return raw
        return self.extrapolator(X, raw)

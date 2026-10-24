"""Serve a frozen model without scikit-learn or scipy.

The desktop app loads ``models/model.pkl``, which holds live sklearn estimators
and scipy interpolators. Android can't install those versions (Chaquopy stops at
scikit-learn 1.3.2, and our pickle is 1.8.0), so ``tools/export_model.py`` in the
release folder flattens the fitted model into arrays. This module reads that
file and rebuilds objects with the same small interface ``Predictor`` expects:

    bundle['full'].predict(X)     -> ndarray
    bundle['calib_full'](value)   -> ndarray
    bundle['tail_top'](X, pred)   -> ndarray, .excess(X), .official_bonus(X)
    bundle['bias_correction'](v), bundle['high_lift'](v)

Everything below is plain numpy; the wrapper classes are imported from
``rating.model`` so the maths lives in exactly one place.
"""

from __future__ import annotations

import json

import numpy as np

from .model import BiasCorrection, Calibration, HighLift, TailExtrapolator


class TreeEnsemble:
    """A gradient-boosted tree ensemble, evaluated straight from flat arrays."""

    def __init__(self, feature, threshold, left, right, value, offsets,
                 learning_rate: float, init: float):
        self.feature = feature.astype(np.int32)
        self.threshold = threshold.astype(np.float64)
        self.left = left.astype(np.int32)
        self.right = right.astype(np.int32)
        self.value = value.astype(np.float64)
        self.offsets = offsets.astype(np.int64)
        self._roots = [int(v) for v in self.offsets]
        self.learning_rate = float(learning_rate)
        self.init = float(init)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] <= 4:
            return self._predict_few(X)
        rows = np.arange(X.shape[0])
        total = np.zeros(X.shape[0])
        for start in self.offsets:
            node = np.full(X.shape[0], int(start), dtype=np.int64)
            while True:
                feature = self.feature[node]
                leaf = feature < 0
                if leaf.all():
                    break
                column = np.where(leaf, 0, feature)
                going_left = X[rows, column] <= np.where(leaf, 0.0, self.threshold[node])
                node = np.where(leaf, node, np.where(going_left, self.left[node], self.right[node]))
            total += self.value[node]
        return self.init + self.learning_rate * total

    def _predict_few(self, X) -> np.ndarray:
        """Scalar walk for one or two rows.

        The vectorised path above issues ~4 numpy calls per tree; for a single
        row that overhead dominates and made inference 13x slower than sklearn's
        Cython traversal. A plain Python loop over the same arrays is ~50x
        faster at n<=4, which is the only shape the web UI ever asks for.
        """
        feature, threshold = self.feature, self.threshold
        left, right, value = self.left, self.right, self.value
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            row = X[i].tolist()
            total = 0.0
            for start in self._roots:
                node = start
                while True:
                    name = feature[node]
                    if name < 0:
                        break
                    node = left[node] if row[name] <= threshold[node] else right[node]
                total += value[node]
            out[i] = total
        return self.init + self.learning_rate * out


class RidgeModel:
    """StandardScaler + Ridge, collapsed into (x - mean) / scale @ coef + b."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean = mean.astype(np.float64)
        self.scale = scale.astype(np.float64)
        self.coef = coef.astype(np.float64)
        self.intercept = float(intercept)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept


class PortableHybrid:
    """Mirror of rating.model.HybridModel.predict (tree + clamped ridge)."""

    def __init__(self, base: TreeEnsemble, ridge: RidgeModel, alpha: float = 8.0):
        self.base = base
        self.ridge = ridge
        self.alpha = alpha

    def predict(self, X) -> np.ndarray:
        return self.base.predict(X) + np.clip(self.ridge.predict(X), -2.0, 2.0)


class PchipCurve:
    """scipy's PchipInterpolator, evaluated from its polynomial coefficients.

    scipy stores one cubic per interval (``c`` has shape (4, len(x) - 1), highest
    power first); evaluating that directly reproduces it bit for bit, including
    how it extrapolates past the last breakpoint.
    """

    def __init__(self, x, c):
        self.x = x.astype(np.float64)
        self.c = c.astype(np.float64)

    def __call__(self, value):
        v = np.asarray(value, dtype=float)
        index = np.clip(np.searchsorted(self.x, v) - 1, 0, len(self.x) - 2)
        t = v - self.x[index]
        c = self.c
        return ((c[0, index] * t + c[1, index]) * t + c[2, index]) * t + c[3, index]


def load_bundle(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data['meta']).decode('utf-8'))

    def hybrid(name: str) -> PortableHybrid:
        idx = data[f'{name}__offset']
        return PortableHybrid(
            TreeEnsemble(data[f'{name}__feature'], data[f'{name}__threshold'],
                         data[f'{name}__left'], data[f'{name}__right'],
                         data[f'{name}__value'], idx,
                         float(data[f'{name}__learning_rate']), float(data[f'{name}__init'])),
            RidgeModel(data[f'{name}__mean'], data[f'{name}__scale'],
                       data[f'{name}__coef'], float(data[f'{name}__intercept'])[0]
                       if data[f'{name}__intercept'].ndim else float(data[f'{name}__intercept'])),
            float(data[f'{name}__alpha']))

    def curve(name: str) -> PchipCurve:
        return PchipCurve(data[f'{name}__x'], data[f'{name}__c'])

    if meta.get('unified'):
        from .extrapolate import BalancedLoadExtrapolator, LoadExtrapolator
        from .unified import ExtrapolatedModel, Identity, ToOfficial, Zeros
        ext_meta = meta['extrapolator']
        if ext_meta.get('type') == 'BalancedLoadExtrapolator':
            ext = BalancedLoadExtrapolator(
                ext_meta['columns'], data['extrapolator__low'], data['extrapolator__high'],
                ext_meta['sustain_columns'], ext_meta['burst_columns'],
                slope=ext_meta['slope'], z_crit=ext_meta['z_crit'],
                soft=ext_meta['soft'], top_k=ext_meta.get('top_k', 3),
                sustain_weight=ext_meta.get('sustain_weight', 0.5))
        else:
            ext = LoadExtrapolator(ext_meta['columns'], data['extrapolator__low'],
                                   data['extrapolator__high'], slope=ext_meta['slope'],
                                   z_crit=ext_meta['z_crit'], soft=ext_meta['soft'],
                                   top_k=ext_meta.get('top_k', 3))
        return {
            'columns': meta['columns'],
            'baseline': meta['baseline'],
            'scale': meta['scale'],
            'blend': (1e9, 1e9 + 1.0),
            'full': ExtrapolatedModel(hybrid('full'), ext),
            'top': Zeros(),
            'calib_full': ToOfficial(Identity()),
            'calib_top': Identity(),
            'tail_top': None,
            'bias_correction': None,
            'high_lift': None,
        }

    bundle = {
        'columns': meta['columns'],
        'baseline': meta['baseline'],
        'scale': meta['scale'],
        'blend': tuple(meta['blend']),
        'full': hybrid('full'),
        'top': hybrid('top'),
        'calib_full': Calibration(curve('calib_full'), meta['calib_full']['low'],
                                  meta['calib_full']['high'], meta['calib_full']['slope_low'],
                                  meta['calib_full']['slope_high']),
        'calib_top': Calibration(curve('calib_top'), meta['calib_top']['low'],
                                 meta['calib_top']['high'], meta['calib_top']['slope_low'],
                                 meta['calib_top']['slope_high']),
        'bias_correction': (None if meta.get('bias_correction') is None else
                            BiasCorrection(curve('bias_correction'),
                                           meta['bias_correction']['damping'],
                                           meta['bias_correction']['limit'],
                                           meta['bias_correction']['center'],
                                           meta['bias_correction']['width'])),
        'tail_top': TailExtrapolator(meta['tail_top']['columns'],
                                     data['tail_top__low'], data['tail_top__high'],
                                     meta['tail_top']['c_max'], meta['tail_top']['rate'],
                                     meta['tail_top']['ceiling']),
        'high_lift': HighLift(meta['high_lift']['start'], meta['high_lift']['target_at'],
                              meta['high_lift']['target_delta'], meta['high_lift']['cap']),
    }
    bundle['high_lift'].k = meta['high_lift']['k']
    return bundle

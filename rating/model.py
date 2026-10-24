"""Train, cross-validate and compare difficulty models on the official scale."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import pickle

import numpy as np
# scikit-learn is only needed to *train*. Keeping the import local lets the
# frozen model be served on devices where sklearn isn't installable (Android),
# because rating.portable imports the classes below without pulling sklearn in.
def _sklearn():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return dict(GradientBoostingRegressor=GradientBoostingRegressor, Ridge=Ridge,
                GroupKFold=GroupKFold, MLPRegressor=MLPRegressor,
                make_pipeline=make_pipeline, StandardScaler=StandardScaler)

from . import scale
from .dataset import ROOT

DATASET = os.path.join(ROOT, 'outputs', 'dataset.pkl')
# 打包版（Android）只有 sklearn-free 的 npz，用环境变量指过去
MODEL_PATH = os.environ.get('PHIGROS_MODEL') or os.path.join(ROOT, 'models', 'model.pkl')

# Features whose absolute value depends on the authoring convention of the
# format rather than on the chart: RPE/PEC scroll speeds are written in a free
# unit that different authors set very differently, so the *level* of the speed
# is only meaningful relative to the rest of the same chart.
LEVEL_SENSITIVE = {'speed_mean', 'speed_p90', 'speed_max', 'throughput', 'throughput_p90'}


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class Result:
    name: str
    mae: float
    bias: float
    per_band: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def load_dataset(path: str = DATASET) -> dict:
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def label_density_weights(labels: np.ndarray, alpha: float = 0.5,
                          bandwidth: float = 0.35) -> np.ndarray:
    """Inverse-density weights: smooth, no difficulty cut-offs.

    ``alpha=0`` is uniform; ``alpha=1`` fully equalises the label distribution.
    A Gaussian kernel on the label axis makes the weight a continuous function of
    the label, so neighbouring difficulties get nearly identical weights (a
    hard-coded "15.0-18.0 counts triple" rule does not have that property).
    """
    if alpha <= 0 or labels.size == 0:
        return np.ones_like(labels, dtype=float)
    delta = labels[:, None] - labels[None, :]
    density = np.exp(-0.5 * (delta / bandwidth) ** 2).sum(axis=1)
    weight = (density / density.mean()) ** (-alpha)
    return weight / weight.mean()


class HybridModel:
    """Gradient boosting plus a smooth linear residual.

    A tree ensemble is piecewise constant: charts that land in the same leaves
    get bit-identical predictions (the "why do so many charts share 17.953"
    problem). Adding a ridge fit on the trees' residual keeps the trees'
    non-linearity while making the output a continuous function of the features,
    so two different charts always get two different numbers.
    """

    def __init__(self, base, alpha: float = 8.0):
        self.base = base
        self.alpha = alpha
        self.ridge = None

    def fit(self, X, y, sample_weight=None):
        sk = _sklearn()
        self.base.fit(X, y, sample_weight=sample_weight)
        residual = y - self.base.predict(X)
        self.ridge = sk['make_pipeline'](sk['StandardScaler'](), sk['Ridge'](alpha=self.alpha))
        weight = {} if sample_weight is None else {'ridge__sample_weight': sample_weight}
        self.ridge.fit(X, residual, **weight)
        return self

    def predict(self, X):
        # The ridge layer exists to make the output continuous, not to carry the
        # prediction: on a chart whose features are far outside the training
        # distribution (a degenerate 0-note parse gives nps -> 1e9) a linear term
        # extrapolates to -400. Clamping it keeps it a refinement.
        return self.base.predict(X) + np.clip(self.ridge.predict(X), -2.0, 2.0)


class Calibration:
    """Monotone OOF calibration with *linear* tails.

    A PCHIP left on ``extrapolate=True`` evaluates its last cubic outside the
    fitted range, and that cubic can turn around: the 19.1-community chart was
    being mapped to 4.4. Outside the fitted range the map must continue with the
    boundary slope instead, which is both monotone and bounded.
    """

    def __init__(self, curve, low: float, high: float, slope_low: float, slope_high: float):
        self.curve = curve
        self.low = float(low)
        self.high = float(high)
        self.slope_low = float(slope_low)
        self.slope_high = float(slope_high)

    def __call__(self, value):
        x = np.asarray(value, dtype=float)
        y = np.asarray(self.curve(np.clip(x, self.low, self.high)), dtype=float)
        y = y + np.where(x > self.high, (x - self.high) * self.slope_high, 0.0)
        y = y + np.where(x < self.low, (x - self.low) * self.slope_low, 0.0)
        return y


class TailExtrapolator:
    """Bounded extension towards the hardest label the data contains.

    A tree cannot predict outside the range of its targets, so ``ex8.json``
    (13131 notes, 23.8 notes/s, tap density 1.5x the training maximum) still
    receives the ceiling value. But the labels themselves flatten out: among the
    31 charts whose load score is in the top 8%, the correlation between load and
    difficulty is only 0.08, and every one of them sits at 18.2-18.7 no matter
    how dense it is. So the honest extension is not "load x slope" — that would
    invent 19.8 for ex8 — but an approach towards the hardest label observed:

        y = y_max - (y_max - y_model) * exp(-k * excess)

    which is smooth, monotone in ``excess``, and can never exceed ``y_max``.
    """

    def __init__(self, columns: list[int], low: np.ndarray, high: np.ndarray,
                 c_max: float, rate: float, ceiling: float):
        self.columns = list(columns)
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.c_max = float(c_max)
        self.rate = float(rate)
        self.ceiling = float(ceiling)

    def composite(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)[:, self.columns]
        span = np.maximum(self.high - self.low, 1e-6)
        return np.clip((values - self.low) / span, 0.0, None).mean(axis=1)

    def excess(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.composite(X) - self.c_max, 0.0)

    def official_bonus(self, X: np.ndarray, slope: float = 0.45,
                       cap: float = 0.7) -> np.ndarray:
        """Official-scale bonus for charts beyond the labelled envelope.

        The community labels stop at 19.3, and the scale map squeezes everything
        above 18.8 into ~0.1 official units, so a chart that is genuinely harder
        than anything labelled still lands at 18.0. This is the explicitly
        expert-anchored part: beyond ``c_max`` each unit of load adds ``slope``
        official units, capped so a single extreme feature cannot run away.
        """
        return np.minimum(self.excess(X) * slope, cap)

    def __call__(self, X: np.ndarray, prediction) -> np.ndarray:
        excess = np.maximum(self.composite(X) - self.c_max, 0.0)
        boost = (self.ceiling - prediction) * (1.0 - np.exp(-self.rate * excess))
        return prediction + np.maximum(boost, 0.0)


# Features that describe "how much the hands have to do". All of them are in
# player units (notes/s), so the composite is comparable across charts, and none
# of them is a degenerate counter that one malformed chart can blow up.
LOAD_FEATURES = ('press_nps', 'finale_d0.5_max', 'finale_press_nps', 'finale_chord_max',
                 'd1_p90', 'nps', 'ch0.05_mean', 'jump_speed_p95')


def fit_tail(X: np.ndarray, y: np.ndarray, names: list[str],
             rate: float = 1.6) -> TailExtrapolator:
    columns = [names.index(name) for name in LOAD_FEATURES if name in names]
    values = X[:, columns]
    low = np.median(values, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    probe = TailExtrapolator(columns, low, high, 0.0, 0.0, 0.0)
    score = probe.composite(X)
    # the extension starts where the labelled evidence starts thinning out
    c_max = float(np.quantile(score, 0.98))
    ceiling = float(np.quantile(y, 0.995))
    return TailExtrapolator(columns, low, high, c_max, rate, ceiling)


class BiasCorrection:
    """Band-level offset, applied only where the reference set has evidence.

    The ranked charts are dense between 14 and 17 and almost absent above 18,
    so a raw residual curve extrapolates hard at the top (it wanted to pull an
    18.98 down by 2.9). The correction is therefore clamped and windowed: full
    strength in 15-17.4, fading out smoothly below 14.5 and above 18.2.
    """

    def __init__(self, curve, damping: float, limit: float = 0.5,
                 center: float = 16.2, width: float = 1.6):
        self.curve = curve
        self.damping = float(damping)
        self.limit = float(limit)
        self.center = float(center)
        self.width = float(width)

    def window(self, x):
        return np.exp(-(((np.asarray(x, dtype=float) - self.center) / self.width) ** 4))

    def __call__(self, value):
        x = np.asarray(value, dtype=float)
        correction = np.clip(np.asarray(self.curve(x), dtype=float), -self.limit, self.limit)
        return x - self.damping * correction * self.window(x)


class HighLift:
    """Smooth lift of the high end, defined on the community scale.

    Charts rated 17.0 or below are left exactly as they are. Above 17.0 the lift
    starts with zero slope (so the curve has no kink at 17.0) and grows
    quadratically, then saturates at ``cap`` so an extreme chart cannot run away:

        q     = k * (c - start)^2
        delta = cap * (1 - exp(-q / cap))

    ``k`` is solved so that ``target_at`` receives exactly ``target_delta``
    (default: 18.8 -> +0.25, i.e. 19.0~19.1). It is a monotone, C1 function of the
    model's own score, so the ranking of charts never changes and no new steps are
    introduced with respect to the features.
    """

    def __init__(self, start: float = 17.0, target_at: float = 18.8,
                 target_delta: float = 0.25, cap: float = 0.5):
        if not 0.0 < target_delta < cap:
            raise ValueError('target_delta must lie in (0, cap)')
        self.start = float(start)
        self.target_at = float(target_at)
        self.target_delta = float(target_delta)
        self.cap = float(cap)
        # invert delta(q) = target_delta for q, then q = k * span^2
        q = -self.cap * np.log(1.0 - self.target_delta / self.cap)
        self.k = float(q / (self.target_at - self.start) ** 2)

    def delta(self, community):
        c = np.asarray(community, dtype=float)
        q = self.k * np.square(np.maximum(c - self.start, 0.0))
        return self.cap * (1.0 - np.exp(-q / self.cap))

    def community(self, community):
        c = np.asarray(community, dtype=float)
        return c + self.delta(c)

    def __call__(self, official):
        """official -> official (the lift is applied on the community scale)."""
        c = scale.official_to_community(official)
        return scale.community_to_official(self.community(c))


def build_model(kind: str):
    if kind == 'ridge':
        sk = _sklearn()
        return sk['make_pipeline'](sk['StandardScaler'](), sk['Ridge'](alpha=3.0))
    if kind == 'gbr':
        sk = _sklearn()
        return sk['GradientBoostingRegressor'](
            n_estimators=500, learning_rate=0.04, max_depth=3,
            subsample=0.85, min_samples_leaf=6, random_state=0)
    if kind == 'mlp':
        sk = _sklearn()
        return sk['make_pipeline'](
            sk['StandardScaler'](),
            sk['MLPRegressor'](hidden_layer_sizes=(48, 16), activation='tanh', alpha=1e-3,
                         learning_rate_init=3e-3, max_iter=4000, early_stopping=True,
                         n_iter_no_change=60, random_state=0))
    raise ValueError(kind)


def grouped_oof(kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                weights: np.ndarray, folds: int = 5) -> np.ndarray:
    predictions = np.full(y.shape, np.nan)
    splitter = _sklearn()['GroupKFold'](n_splits=folds)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = build_model(kind)
        model.fit(X[train_idx], y[train_idx], **_fit_kwargs(model, weights[train_idx]))
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions


def _fit_kwargs(model, sample_weight):
    """Pass sample_weight to the estimator (pipelines need the step name)."""
    if isinstance(sample_weight, np.ndarray) and np.allclose(sample_weight, 1.0):
        return {}
    if hasattr(model, 'fit') and model.__class__.__name__ in ('Ridge', 'GradientBoostingRegressor'):
        return {'sample_weight': sample_weight}
    if hasattr(model, 'steps'):
        name = model.steps[-1][0]
        return {f'{name}__sample_weight': sample_weight}
    return {}


def score(y: np.ndarray, pred: np.ndarray) -> Result:
    err = pred - y
    result = Result(name='', mae=float(np.abs(err).mean()), bias=float(err.mean()))
    bands = [(0, 12), (12, 14), (14, 15.5), (15.5, 17), (17, 18.1)]
    for low, high in bands:
        mask = (y >= low) & (y < high)
        if mask.sum():
            result.per_band[f'{low}-{high}'] = (
                int(mask.sum()), float(np.abs(err[mask]).mean()))
    return result


def fmt(result: Result) -> str:
    bands = ' '.join(f'{k}:{v[1]:.3f}({v[0]})' for k, v in result.per_band.items())
    return f'{result.name:<28} MAE {result.mae:.4f}  bias {result.bias:+.4f}  | {bands}'

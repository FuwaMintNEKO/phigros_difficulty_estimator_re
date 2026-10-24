"""Bounded extrapolation beyond the training envelope.

A tree ensemble cannot predict outside the range of its targets, so every chart
denser than the training maximum collapses onto the same leaf. That is fine for
charts inside the envelope and wrong for the ones deliberately made to break it
(4K spam well past anything Phigros ships).

The extension is the same shape the old pipeline used, but it now runs on the
unified community scale and reads several dimensions instead of density alone:

    e_f    = max((x_f - median_f) / (p99_f - median_f) - 1, 0)   per feature
    excess = max(mean(top-k of e_f) - c_max, 0)
    y      = y_max - (y_max - y) * exp(-rate * excess)

Averaging the normalised values (the old design) cancels out the signal: a 4K
chart is brutal on density but has little lane movement, so the low movement
terms wash out the density ones. Taking the mean of the *top-k exceedances*
instead means only the dimensions that actually left the envelope speak, and a
chart that leaves it in several dimensions accumulates.
"""

from __future__ import annotations

import numpy as np

# 每个维度各取一两个代表量，避免“只看密度”。
# 密度：需要按下的键；爆发：短窗口极值；多押：同时按下的数；位移：跳跃速度；体力：持续段。
# nps 与 press_nps 现在是同一个量（密度统一按 tap+hold），只留一个，避免重复计权
LOAD_FEATURES = (
    'nps', 'pressure_load', 'd1_p90',
    'ch0.02_mean', 'ch0.02_max',
    'd0.5_max', 'd1_max', 'jump_speed_p95',
)
# 刻意不放 speed_p90 / hold_coverage：前者是谱师的滚动速度偏好（与难度无关，
# 实测有低难谱 speed_p90 超出训练 p99 达 5 倍），后者在短谱上噪声极大。


class LoadExtrapolator:
    def __init__(self, columns, low, high, c_max: float = 0.0, rate: float = 0.0,
                 ceiling: float = 0.0, top_k: int = 3, slope: float = 0.0,
                 z_crit: float = 1.0, soft: float = 0.15):
        self.columns = list(columns)
        self.top_k = int(top_k)
        self.slope = float(slope)
        self.z_crit = float(z_crit)
        self.soft = float(soft)
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.c_max = float(c_max)
        self.rate = float(rate)
        self.ceiling = float(ceiling)

    def position(self, X) -> np.ndarray:
        """每个特征在 [中位数, p99] 上的归一化位置，取最强的几维平均。"""
        values = np.asarray(X, dtype=float)[:, self.columns]
        span = np.maximum(self.high - self.low, 1e-6)
        scaled = np.clip((values - self.low) / span, 0.0, None)
        k = min(self.top_k, scaled.shape[1])
        top = -np.sort(-scaled, axis=1)[:, :k]
        return top.mean(axis=1)

    def per_feature_excess(self, X) -> np.ndarray:
        values = np.asarray(X, dtype=float)[:, self.columns]
        span = np.maximum(self.high - self.low, 1e-6)
        return np.clip((values - self.low) / span - 1.0, 0.0, None)

    def composite(self, X) -> np.ndarray:
        excess = self.per_feature_excess(X)
        if not excess.size:
            return np.zeros(len(excess))
        k = min(self.top_k, excess.shape[1])
        top = -np.sort(-excess, axis=1)[:, :k]
        return top.mean(axis=1)

    def excess(self, X) -> np.ndarray:
        return np.maximum(self.composite(X) - self.c_max, 0.0)

    def __call__(self, X, prediction) -> np.ndarray:
        """从临界点 z_crit 起，按实测斜率线性延长，并做平滑接入。

        d = z - z_crit；接入项 = slope * d^2 / (d + soft)，在 d=0 处值和斜率都为 0，
        d 变大后趋近 slope * d（线性）。所以包络内的谱完全不受影响。
        """
        d = np.maximum(self.position(X) - self.z_crit, 0.0)
        ramp = d * d / (d + self.soft)
        return np.asarray(prediction, dtype=float) + self.slope * ramp


class BalancedLoadExtrapolator(LoadExtrapolator):
    """Use separate sustained-pressure and burst groups for tail extrapolation.

    A single top-k pool lets a short multi-press tail dominate a chart whose
    entire body is harder. The groups are intentionally small and semantic:
    sustained density includes the player-facing pressure integral; burst keeps
    the short-window and tail peaks. ``sustain_weight`` is applied only to the
    extrapolation score, not to tree training.
    """

    def __init__(self, columns, low, high, sustain_columns, burst_columns,
                 slope, z_crit, soft=0.15, top_k=2, sustain_weight=0.65):
        super().__init__(columns, low, high, slope=slope, z_crit=z_crit,
                         soft=soft, top_k=top_k)
        self.sustain_columns = list(sustain_columns)
        self.burst_columns = list(burst_columns)
        self.sustain_weight = float(sustain_weight)

    def _group_position(self, X, columns):
        values = np.asarray(X, dtype=float)[:, columns]
        low = self.low[[self.columns.index(c) for c in columns]]
        high = self.high[[self.columns.index(c) for c in columns]]
        scaled = np.clip((values - low) / np.maximum(high - low, 1e-6), 0.0, None)
        k = min(self.top_k, scaled.shape[1])
        return -np.sort(-scaled, axis=1)[:, :k].mean(axis=1)

    def position(self, X):
        sustained = self._group_position(X, self.sustain_columns)
        burst = self._group_position(X, self.burst_columns)
        return self.sustain_weight * sustained + (1.0 - self.sustain_weight) * burst


def fit_balanced_extrapolator(X, names: list[str], y, *,
                              sustain_weight: float = 0.65,
                              head_lo: float = 0.35) -> BalancedLoadExtrapolator | None:
    sustain_names = ('pressure_load', 'press_nps', 'd1_p90', 'finale_press_nps')
    burst_names = ('d0.5_max', 'finale_d0.5_max', 'd1_max', 'ch0.1_max')
    chosen = [n for n in sustain_names + burst_names if n in names]
    if len(chosen) < 4:
        return None
    columns = [names.index(n) for n in chosen]
    values = np.asarray(X, dtype=float)[:, columns]
    low = np.median(values, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    sustain_cols = [names.index(n) for n in sustain_names if n in names]
    burst_cols = [names.index(n) for n in burst_names if n in names]
    probe = BalancedLoadExtrapolator(columns, low, high, sustain_cols, burst_cols,
                                     slope=0.0, z_crit=0.0,
                                     sustain_weight=sustain_weight)
    z = probe.position(X)
    upper = z > head_lo
    design = np.column_stack([z[upper], np.ones(int(upper.sum()))])
    slope, _ = np.linalg.lstsq(design, np.asarray(y, dtype=float)[upper], rcond=None)[0]
    return BalancedLoadExtrapolator(columns, low, high, sustain_cols, burst_cols,
                                    slope=float(max(slope, 0.0)), z_crit=float(z.max()),
                                    sustain_weight=sustain_weight)


def fit_extrapolator(X, names: list[str], y, *, head_lo: float = 0.35) -> LoadExtrapolator | None:
    columns = [names.index(name) for name in LOAD_FEATURES if name in names]
    if len(columns) < 4:
        return None
    values = np.asarray(X, dtype=float)[:, columns]
    low = np.median(values, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    probe = LoadExtrapolator(columns, low, high)
    z = probe.position(X)
    # 上段线性拟合：z > head_lo 的样本决定"每单位负荷值多少定数"
    upper = z > head_lo
    design = np.column_stack([z[upper], np.ones(int(upper.sum()))])
    slope, intercept = np.linalg.lstsq(design, np.asarray(y, dtype=float)[upper], rcond=None)[0]
    return LoadExtrapolator(columns, low, high, slope=float(max(slope, 0.0)),
                            z_crit=float(z.max()), soft=0.15)

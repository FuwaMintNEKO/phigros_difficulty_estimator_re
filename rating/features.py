"""Feature extraction for chart difficulty.

Design rules
------------
* Every feature is expressed in a unit a player would recognise (notes per
  second, screen heights per second, ratios). Nothing is a raw counter that
  changes meaning with chart length.
* Nothing is compared against a hand-picked threshold. Density is measured with
  box kernels of several half-widths, simultaneity with several time radii, and
  hand balance with the continuous weight ``(1 - x) / 2`` instead of a left/right
  split. The result is that two charts differing by a tiny amount produce nearly
  identical features, which is what keeps predictions smooth.
* Stats are quantiles (p50/p90/p99/max) rather than counts above a cutoff, so a
  single extra note moves the feature by a proportionate amount.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .charts import ChartData, TYPE_DRAG, TYPE_FLICK, TYPE_HOLD, TYPE_TAP
from .pressure import pressure_summary

# Half-widths (seconds) of the box kernels used for density curves. The set is
# geometric so the model sees the same shape at every time scale.
DENSITY_HALVES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# 多判定线：两个 Tap/Hold 的最大允许间隔（秒）。0.10 = 相隔 100ms 内算同一次多线击打
MULTILINE_WINDOWS = (0.10, 0.20)
# Radii (seconds) for "how many other notes are within reach".
CHORD_RADII = (0.02,)   # 只保留真和弦；0.05/0.1/0.2 实际是短段密度，与 d* 重复


def _density(times: np.ndarray, duration: float, half: float):
    step = float(np.clip(half / 4.0, 0.02, 0.25))
    grid = np.arange(0.0, max(duration, step) + step, step)
    lo = np.searchsorted(times, grid - half, side='left')
    hi = np.searchsorted(times, grid + half, side='right')
    return (hi - lo) / (2.0 * half)


def _weighted_density(times: np.ndarray, weights: np.ndarray, duration: float,
                      half: float):
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    step = float(np.clip(half / 4.0, 0.02, 0.25))
    grid = np.arange(0.0, max(duration, step) + step, step)
    lo = np.searchsorted(times, grid - half, side='left')
    hi = np.searchsorted(times, grid + half, side='right')
    return (cumulative[hi] - cumulative[lo]) / (2.0 * half)


def _stats(curve: np.ndarray) -> dict:
    if curve.size == 0:
        return {'max': 0.0, 'mean': 0.0, 'p90': 0.0, 'p99': 0.0}
    return {
        'max': float(curve.max()),
        'mean': float(curve.mean()),
        'p90': float(np.quantile(curve, 0.90)),
        'p99': float(np.quantile(curve, 0.99)),
    }


def _hold_coverage(chart: ChartData) -> float:
    """Fraction of the chart's timeline with at least one Hold active."""
    mask = chart.types == TYPE_HOLD
    if not mask.any() or chart.duration <= 0:
        return 0.0
    spans = np.stack([chart.times[mask], chart.hold_ends[mask]], axis=1)
    spans.sort(axis=0)
    covered, cursor = 0.0, 0.0
    for start, end in spans:
        start = max(start, cursor)
        if end > start:
            covered += end - start
            cursor = end
    return float(covered / max(chart.duration, 1e-9))


def _max_concurrent_holds(chart: ChartData) -> int:
    mask = chart.types == TYPE_HOLD
    if not mask.any():
        return 0
    events = []
    for t, e in zip(chart.times[mask], chart.hold_ends[mask]):
        events.append((t, 1))
        events.append((e, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    live = best = 0
    for _, delta in events:
        live += delta
        best = max(best, live)
    return int(best)


def extract(chart: ChartData) -> dict[str, float]:
    """Return an ordered dict of features for one chart.

    只统计真实按下的音符：Tap 的点击与 Hold 的头（``press``）。Drag / Flick
    不吃按键，假音符已在解析阶段剔除，因此密度、多押、手部、位移、节奏
    全部建立在同一条 press 时间线上。
    """
    f: dict[str, float] = {}
    types = chart.types
    press_mask = ((types == TYPE_TAP) | (types == TYPE_HOLD)) & ~chart.fakes
    t = chart.times[press_mask]
    px = chart.xs[press_mask]
    sp = np.clip(chart.speeds[press_mask], 0.0, 50.0)
    n = int(t.size)
    duration = max(chart.duration, 1e-9)

    f['n_notes'] = float(chart.times.size)      # 全部真实音符（含 drag/flick），展示用
    f['n_press'] = float(n)
    f['duration'] = float(duration)
    f['nps'] = float(n / duration)
    active = float(t[-1] - t[0]) if n > 1 else duration
    f['nps_active'] = float(n / max(active, 1e-9))
    f['notes_per_minute'] = f['nps'] * 60.0
    f['press_nps'] = f['nps']
    f['pressure_load'] = pressure_summary(chart)['pressure_load']
    if n <= 2:
        return f

    # ----- density at six time scales ----------------------------------
    curves = {}
    for half in DENSITY_HALVES:
        curve = _density(t, duration, half)
        curves[half] = curve
        stats = _stats(curve)
        tag = f'd{half:g}'
        f[f'{tag}_max'] = stats['max']
        f[f'{tag}_p90'] = stats['p90']
        f[f'{tag}_p99'] = stats['p99']
        f[f'{tag}_peak_ratio'] = stats['max'] / max(stats['mean'], 1e-9)
    f['d1_mean'] = f['nps']

    f['burst_1s_over_mean'] = (f['d1_max'] + 0.1) / (f['d1_mean'] + 0.1)
    f['sustain_4s_over_1s'] = (f['d4_p90'] + 0.1) / (f['d1_p90'] + 0.1)
    f['peak_sharpness'] = (f['d0.5_max'] + 0.1) / (f['d8_p90'] + 0.1)

    d1_curve = curves[1.0]
    if d1_curve.size:
        f['sustain_frac'] = float((d1_curve > f['d1_mean']).mean())
        f['sustain_flatness'] = float(f['d1_p90'] / (f['d1_mean'] + 1e-9))
        f['sustain_floor'] = float(np.quantile(d1_curve, 0.25) / (f['d1_mean'] + 1e-9))

    quarter = max(len(d1_curve) // 4, 1)
    f['trend_late_early'] = float((d1_curve[-quarter:].mean() + 0.1)
                                  / (d1_curve[:quarter].mean() + 0.1))
    f['trend_slope'] = float(np.polyfit(np.linspace(0, 1, d1_curve.size), d1_curve, 1)[0]) \
        if d1_curve.size > 2 else 0.0

    # ----- structure: bursts / rest / travel ----------------------------
    if d1_curve.size > 4:
        top = np.quantile(d1_curve, 0.90)
        above = d1_curve > top
        starts = np.count_nonzero(above & ~np.r_[False, above[:-1]])
        f['burst_count_per_min'] = float(starts / duration * 60.0)
        f['rest_ratio'] = float((d1_curve < 0.5 * max(f['d1_mean'], 1e-9)).mean())
        share = d1_curve / max(d1_curve.sum(), 1e-9)
        f['density_entropy'] = float(-(share * np.log(share + 1e-12)).sum()
                                     / np.log(max(d1_curve.size, 2)))
    else:
        f['burst_count_per_min'] = f['rest_ratio'] = f['density_entropy'] = 0.0
    f['hand_travel_rate'] = float(np.abs(np.diff(px)).sum() / duration) if n > 1 else 0.0

    # ----- simultaneity / chords ---------------------------------------
    for radius in CHORD_RADII:
        lo = np.searchsorted(t, t - radius, side='left')
        hi = np.searchsorted(t, t + radius, side='right')
        neighbours = (hi - lo - 1).astype(float)
        tag = f'ch{radius:g}'
        f[f'{tag}_mean'] = float(neighbours.mean())
        f[f'{tag}_p95'] = float(np.quantile(neighbours, 0.95))
        f[f'{tag}_max'] = float(neighbours.max())
        f[f'{tag}_solo'] = float((neighbours == 0).mean())
        sizes, counts = np.unique(neighbours, return_counts=True)
        share = counts / counts.sum()
        f[f'{tag}_entropy'] = float(-(share * np.log(share)).sum())

    # ----- hands -------------------------------------------------------
    left_weight = (1.0 - px) / 2.0
    f['hand_left_frac'] = float(left_weight.mean())
    f['hand_balance'] = float(abs(left_weight.mean() - 0.5) * 2.0)

    same = []
    for radius in (0.05, 0.15):
        lo = np.searchsorted(t, t - radius, side='left')
        hi = np.searchsorted(t, t + radius, side='right')
        spread = np.zeros(n)
        for i in range(n):
            if hi[i] - lo[i] > 1:
                window = px[lo[i]:hi[i]]
                spread[i] = window.max() - window.min()
        same.append(spread)
    f['spread_50ms_mean'] = float(same[0].mean())
    f['spread_50ms_p95'] = float(np.quantile(same[0], 0.95))
    f['spread_150ms_mean'] = float(same[1].mean())

    # ----- movement ----------------------------------------------------
    dt = np.diff(t)
    dx = np.abs(np.diff(px))
    dt_safe = np.maximum(dt, 0.01)
    close = dt > 1e-6
    if close.any():
        f['jump_mean'] = float(dx[close].mean())
        f['jump_close_mean'] = float(dx[close & (dt < 0.2)].mean()) if (close & (dt < 0.2)).any() else 0.0
        f['jump_speed_mean'] = float((dx[close] / dt_safe[close]).mean())
        f['jump_speed_p95'] = float(np.quantile(dx[close] / dt_safe[close], 0.95))
        cross = (1.0 - px[:-1] * px[1:]) / 2.0
        weight = np.exp(-np.clip(dt, 0, None) / 0.15)
        f['alternation'] = float((cross * weight)[close].sum() / max(weight[close].sum(), 1e-9))
    else:
        f['jump_mean'] = f['jump_close_mean'] = 0.0
        f['jump_speed_mean'] = f['jump_speed_p95'] = f['alternation'] = 0.0

    # ----- scroll speed -------------------------------------------------
    f['speed_mean'] = float(sp.mean())
    f['speed_median'] = float(np.median(sp))
    f['speed_p90'] = float(np.quantile(sp, 0.90))
    f['speed_log_p90'] = float(np.log1p(f['speed_p90']))
    f['speed_p90_over_mean'] = (f['speed_p90'] + 0.5) / (f['speed_mean'] + 0.5)
    f['notes_per_height'] = f['nps'] / max(f['speed_mean'], 0.05)
    f['throughput'] = f['nps'] * f['speed_mean']
    f['throughput_p90'] = f['d1_p90'] * f['speed_p90']

    # ----- flash arrival（闪现）------------------------------------------
    # 流速越快，音符从屏幕外进入的可读时间越短。fall = 2 / speed（屏高/秒）。
    # 两套平滑核：0.35 宽松、0.15 更陡，用于比较哪套更有正收益。
    fall = 2.0 / np.maximum(sp, 0.05)
    # 先扣掉“正常速度”的基线：下落 ≥0.8s 的部分不贡献分数，
    # 只有真正看不出落过程（fall 很短）才计分。
    base = float(np.exp(-0.8 / 0.35))
    f['flash_load'] = float(np.maximum(np.exp(-fall / 0.35) - base, 0.0).mean() / (1.0 - base))

    # ----- holds --------------------------------------------------------
    hold_mask = chart.types[press_mask] == TYPE_HOLD
    f['hold_frac'] = float(hold_mask.mean())
    f['hold_coverage'] = _hold_coverage(chart)
    # 按住一条 Hold 期间还要按几个键，按 hold 时长加权平均
    if hold_mask.any():
        starts = t[hold_mask]
        ends = chart.hold_ends[press_mask][hold_mask]
        lo = np.searchsorted(t, starts, side='right')
        hi = np.searchsorted(t, ends, side='left')
        inside = np.maximum(hi - lo, 0).astype(float)
        spans = np.maximum(ends - starts, 0.0)
        f['hold_combo'] = float((inside * spans).sum() / spans.sum()) if spans.sum() > 0 else 0.0
    else:
        f['hold_combo'] = 0.0
    if hold_mask.any():
        lengths = chart.hold_ends[press_mask][hold_mask] - t[hold_mask]
        f['hold_len_mean'] = float(lengths.mean())
        f['hold_len_p90'] = float(np.quantile(lengths, 0.90))
        f['hold_len_max'] = float(lengths.max())
        hold_fall = fall[hold_mask]
        f['flash_hold_load'] = float(np.maximum(np.exp(-hold_fall / 0.35) - base, 0.0).mean()
                                     / (1.0 - base))
    else:
        f['hold_len_mean'] = f['hold_len_p90'] = f['hold_len_max'] = 0.0
        f['flash_hold_load'] = 0.0

    # jackness 已移除：它只比较音符的局部 x，跨判定线时坐标不可比（一张谱可有 58 条线），
    # 会把“不同线上的同一 x”误判成同位置连打。

    # ----- finale (尾杀) -------------------------------------------------
    tail_start = float(chart.times[-1]) - max(0.25 * duration, 20.0)
    tail_times = t[t >= tail_start]
    if tail_times.size >= 8:
        span = max(float(chart.times[-1]) - tail_start, 1.0)
        for half in (0.25, 0.5, 1.0):
            curve = _density(tail_times, float(chart.times[-1]), half)
            f[f'finale_d{half:g}_max'] = float(curve.max()) if curve.size else 0.0
        order = np.searchsorted(tail_times, tail_times - 0.02, side='left')
        order2 = np.searchsorted(tail_times, tail_times + 0.02, side='right')
        f['finale_chord_max'] = float((order2 - order - 1).max())
        f['finale_press_nps'] = float(tail_times.size / span)
    else:
        f['finale_d0.25_max'] = f['finale_d0.5_max'] = f['finale_d1_max'] = 0.0
        f['finale_chord_max'] = f['finale_press_nps'] = 0.0

    # ----- hand-locking and jacks --------------------------------------
    if close.any():
        same_place = np.exp(-dx[close] / 0.06)
        quick = np.exp(-dt_safe[close] / 0.12)
        f['jackness'] = float((same_place * quick).mean())
        f['jackness_p95'] = float(np.quantile(same_place * quick, 0.95))
    else:
        f['jackness'] = f['jackness_p95'] = 0.0

    # ----- note type mix (只在 press 内部区分 Tap / Hold) -----------------
    f['frac_tap'] = float((chart.types[press_mask] == TYPE_TAP).mean())
    f['frac_below'] = float((~chart.above[press_mask]).mean())

    # ----- per-hand load over time --------------------------------------
    left_curve = _weighted_density(t, left_weight, duration, 1.0)
    right_curve = _weighted_density(t, 1.0 - left_weight, duration, 1.0)
    total_curve = left_curve + right_curve
    f['hand_imbalance_p95'] = float(np.quantile(
        np.abs(left_curve - right_curve) / np.maximum(total_curve, 1e-9), 0.95))
    f['hand_left_peak'] = float(left_curve.max())
    f['hand_right_peak'] = float(right_curve.max())

    # ----- multi judge-line play ---------------------------------------
    # 两条判定线上的按键在 2W 内交替，才算“多线击打”。只统计 Tap/Hold。
    if chart.lines is not None and n > 2:
        line_ids = np.asarray(chart.lines, dtype=int)[press_mask]
        for W in MULTILINE_WINDOWS:
            half = W          # W = 两个音符允许的最大间隔
            lo = np.searchsorted(t, t - half, side='left')
            hi = np.searchsorted(t, t + half, side='right')
            flagged = np.zeros(n, dtype=bool)
            lines_in_window = np.zeros(n, dtype=float)
            for i in range(n):
                window = line_ids[lo[i]:hi[i]]
                if window.size > 1:
                    distinct = len(np.unique(window))
                    lines_in_window[i] = distinct
                    flagged[i] = distinct >= 2
            tag = f'ml{W:g}'
            f[f'{tag}_notes'] = float(flagged.sum())
            f[f'{tag}_note_frac'] = float(flagged.mean())
            f[f'{tag}_lines'] = float(lines_in_window[flagged].mean()) if flagged.any() else 0.0
            if flagged.any():
                spans = np.stack([np.clip(t[flagged] - half, 0, duration),
                                  np.clip(t[flagged] + half, 0, duration)], axis=1)
                spans.sort(axis=0)
                covered, cursor = 0.0, 0.0
                for start, end in spans:
                    start = max(start, cursor)
                    if end > start:
                        covered += end - start
                        cursor = end
            else:
                covered = 0.0
            f[f'{tag}_seconds'] = float(covered)
            f[f'{tag}_frac'] = float(covered / duration)
            f[f'{tag}_density'] = float(flagged.sum() / covered) if covered > 1e-9 else 0.0
        f['ml_weighted_frac'] = 0.2 * f['ml0.1_frac'] + 0.8 * f['ml0.2_frac']
        f['ml_weighted_density'] = 0.2 * f['ml0.1_density'] + 0.8 * f['ml0.2_density']

    # ----- judge-line distribution --------------------------------------
    # 有的谱（CREDITS、Desultory）不靠“同一瞬间两条线”，而是音符散在多条线上，
    # 读谱时要同时盯多条轨道。用分布量刻画，避免只看同时性。
    if chart.lines is not None and n > 2:
        ids = np.asarray(chart.lines, dtype=int)[press_mask]
        values, counts = np.unique(ids, return_counts=True)
        share = counts / counts.sum()
        f['line_count'] = float(values.size)
        f['off_main_line_frac'] = float(1.0 - share.max())
        f['line_entropy'] = float(-(share * np.log(share)).sum())
        f['line_cross_rate'] = float((ids[1:] != ids[:-1]).mean())

    # ----- type switch: tap/hold 后面接 drag/flick ----------------------
    # 这是 drag/flick 真正吃读谱的地方；flick 权重远高于 drag。
    raw_t = np.asarray(chart.times, dtype=float)
    raw_ty = np.asarray(chart.types, dtype=int)
    if raw_t.size > 2:
        gap = np.diff(raw_t)
        prev_press = (raw_ty[:-1] == TYPE_TAP) | (raw_ty[:-1] == TYPE_HOLD)
        nxt_flick = raw_ty[1:] == TYPE_FLICK
        nxt_drag = raw_ty[1:] == TYPE_DRAG
        decay = np.exp(-np.clip(gap, 0, None) / 0.30)
        sel_flick = prev_press & nxt_flick & (gap > 1e-6)
        sel_drag = prev_press & nxt_drag & (gap > 1e-6)
        flick_w = float(decay[sel_flick].sum())
        drag_w = float(decay[sel_drag].sum())
        f['type_switch_load'] = (flick_w + 0.05 * drag_w) / duration
        f['type_switch_flick_rate'] = flick_w / duration
        f['type_switch_drag_rate'] = drag_w / duration
    else:
        f['type_switch_load'] = f['type_switch_flick_rate'] = f['type_switch_drag_rate'] = 0.0

    # ----- rhythm regularity -------------------------------------------
    positive = dt[dt > 1e-6]
    if positive.size > 4:
        ratio = positive / max(np.median(positive), 1e-9)
        f['interval_cv'] = float(positive.std() / max(positive.mean(), 1e-9))
        f['rhythm_regularity'] = float(np.median(np.abs(np.diff(np.log(ratio)))) if ratio.size > 4 else 0.0)
        f['grid_concentration'] = float(np.mean(np.abs(ratio - 1.0) < 0.1))
        f['dt_p10'] = float(np.quantile(positive, 0.10))
        f['dt_p25'] = float(np.quantile(positive, 0.25))
        f['dt_median'] = float(np.median(positive))
        f['fast_frac'] = float(np.exp(-positive / 0.08).mean())
        if positive.size > 2:
            second = px[2:] - 2.0 * px[1:-1] + px[:-2]
            f['x_jagged_mean'] = float((np.abs(second) / dt_safe[:-1]).mean())
            f['x_jagged_p95'] = float(np.quantile(np.abs(second) / dt_safe[:-1], 0.95))
        else:
            f['x_jagged_mean'] = f['x_jagged_p95'] = 0.0
    else:
        f['interval_cv'] = f['rhythm_regularity'] = f['grid_concentration'] = 0.0
        f['dt_p10'] = f['dt_p25'] = f['dt_median'] = f['fast_frac'] = 0.0
        f['x_jagged_mean'] = f['x_jagged_p95'] = 0.0

    return f


FEATURE_VERSION = 'v12'

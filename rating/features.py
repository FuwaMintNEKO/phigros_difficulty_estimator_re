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

# Half-widths (seconds) of the box kernels used for density curves. The set is
# geometric so the model sees the same shape at every time scale.
DENSITY_HALVES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Radii (seconds) for "how many other notes are within reach".
CHORD_RADII = (0.02, 0.05, 0.1, 0.2)


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
    """Return an ordered dict of features for one chart."""
    f: dict[str, float] = {}
    # Fake notes need no input at all - they are decoration. They were only being
    # counted (as frac_fake) while still inflating every density, chord and finale
    # feature; one chart in the sample is 28.8% fakes. Record the fraction, then
    # drop them from everything else.
    f['frac_fake'] = float(chart.fakes.mean()) if chart.fakes.size else 0.0
    if chart.fakes.any():
        keep = ~chart.fakes
        chart = replace(chart, times=chart.times[keep], xs=chart.xs[keep],
                        types=chart.types[keep], above=chart.above[keep],
                        fakes=chart.fakes[keep], hold_ends=chart.hold_ends[keep],
                        speeds=chart.speeds[keep])
    times = chart.times
    n = times.size
    duration = max(chart.duration, 1e-9)
    # A handful of charts contain a degenerate scroll event (an author typing a
    # huge value); 50 screen heights per second is already twenty times faster
    # than any playable chart, so clamp there instead of letting one event
    # dominate every speed-derived feature.
    speeds = np.clip(chart.speeds, 0.0, 50.0)
    # Effective load is what the fingers must *press*: Tap needs a press and so
    # does the head of a Hold. Drag and Flick are contact/swipe notes, and a pile
    # of them at one instant is visual noise, not difficulty - counting them made
    # the density peaks of ordinary charts (74/s) look harder than a real 32nd
    # burst (38/s). Density, chords and the finale therefore use press notes only.
    press_mask = (chart.types == TYPE_TAP) | (chart.types == TYPE_HOLD)
    press_times = times[press_mask] if press_mask.any() else times
    press_xs = chart.xs[press_mask] if press_mask.any() else chart.xs

    # ----- global size -------------------------------------------------
    f['n_notes'] = float(n)
    f['duration'] = float(duration)
    f['nps'] = float(n / duration)
    active = float(times[-1] - times[0]) if n > 1 else duration
    f['nps_active'] = float(n / max(active, 1e-9))
    f['notes_per_minute'] = f['nps'] * 60.0
    if n <= 2:
        return f

    # ----- density at six time scales ----------------------------------
    curves = {}
    for half in DENSITY_HALVES:
        curve = _density(press_times, duration, half)
        curves[half] = curve
        stats = _stats(curve)
        tag = f'd{half:g}'
        f[f'{tag}_max'] = stats['max']
        f[f'{tag}_p90'] = stats['p90']
        f[f'{tag}_p99'] = stats['p99']
        f[f'{tag}_peak_ratio'] = stats['max'] / max(stats['mean'], 1e-9)
    # The mean of a box-density curve is n/duration for every window, so it is
    # kept once instead of once per scale.
    f['d1_mean'] = float(press_times.size / duration)

    # burst / stamina, both as ratios of the same unit (notes per second)
    f['burst_1s_over_mean'] = (f['d1_max'] + 0.1) / (f['d1_mean'] + 0.1)
    f['sustain_4s_over_1s'] = (f['d4_p90'] + 0.1) / (f['d1_p90'] + 0.1)
    f['peak_sharpness'] = (f['d0.5_max'] + 0.1) / (f['d8_p90'] + 0.1)

    # density trend: is the second half harder than the first?
    quarter = max(len(curves[1.0]) // 4, 1)
    d1 = curves[1.0]
    f['trend_late_early'] = float((d1[-quarter:].mean() + 0.1) / (d1[:quarter].mean() + 0.1))
    f['trend_slope'] = float(np.polyfit(np.linspace(0, 1, d1.size), d1, 1)[0]) \
        if d1.size > 2 else 0.0

    # ----- simultaneity / chords ---------------------------------------
    for radius in CHORD_RADII:
        lo = np.searchsorted(press_times, press_times - radius, side='left')
        hi = np.searchsorted(press_times, press_times + radius, side='right')
        neighbours = (hi - lo - 1).astype(float)
        tag = f'ch{radius:g}'
        f[f'{tag}_mean'] = float(neighbours.mean())
        f[f'{tag}_p95'] = float(np.quantile(neighbours, 0.95))
        f[f'{tag}_max'] = float(neighbours.max())
        f[f'{tag}_solo'] = float((neighbours == 0).mean())

    # ----- hands -------------------------------------------------------
    left_weight = (1.0 - chart.xs) / 2.0          # 1 at x=-1, 0 at x=+1
    f['hand_left_frac'] = float(left_weight.mean())
    f['hand_balance'] = float(abs(left_weight.mean() - 0.5) * 2.0)

    # same-instant spread across the screen
    same = []
    for radius in (0.05, 0.15):
        lo = np.searchsorted(times, times - radius, side='left')
        hi = np.searchsorted(times, times + radius, side='right')
        spread = np.zeros(n)
        for i in range(n):
            if hi[i] - lo[i] > 1:
                window = chart.xs[lo[i]:hi[i]]
                spread[i] = window.max() - window.min()
        same.append(spread)
    f['spread_50ms_mean'] = float(same[0].mean())
    f['spread_50ms_p95'] = float(np.quantile(same[0], 0.95))
    f['spread_150ms_mean'] = float(same[1].mean())

    # consecutive-note jumps: small = jack, large = cross-screen jump
    dt = np.diff(times)
    dx = np.abs(np.diff(chart.xs))
    dt_safe = np.maximum(dt, 0.01)          # notes closer than 10 ms are simultaneous
    close = dt > 1e-6
    if close.any():
        f['jump_mean'] = float(dx[close].mean())
        f['jump_close_mean'] = float(dx[close & (dt < 0.2)].mean()) if (close & (dt < 0.2)).any() else 0.0
        f['jump_speed_mean'] = float((dx[close] / dt_safe[close]).mean())
        f['jump_speed_p95'] = float(np.quantile(dx[close] / dt_safe[close], 0.95))
        # alternation: opposite sides alternating quickly
        cross = (1.0 - chart.xs[:-1] * chart.xs[1:]) / 2.0
        weight = np.exp(-np.clip(dt, 0, None) / 0.15)
        f['alternation'] = float((cross * weight)[close].sum() / max(weight[close].sum(), 1e-9))
    else:
        f['jump_mean'] = f['jump_close_mean'] = 0.0
        f['jump_speed_mean'] = f['jump_speed_p95'] = f['alternation'] = 0.0

    # ----- scroll speed (movement) -------------------------------------
    f['speed_mean'] = float(speeds.mean())
    f['speed_median'] = float(np.median(speeds))
    f['speed_p90'] = float(np.quantile(speeds, 0.90))
    f['speed_log_p90'] = float(np.log1p(f['speed_p90']))
    f['speed_p90_over_mean'] = (f['speed_p90'] + 0.5) / (f['speed_mean'] + 0.5)
    # how much vertical distance the player must parse per second
    f['throughput'] = f['nps'] * f['speed_mean']
    f['throughput_p90'] = f['d1_p90'] * f['speed_p90']
    # notes per screen height: visual spacing, the thing that makes a dense
    # chart readable or a wall of notes
    f['notes_per_height'] = f['nps'] / max(f['speed_mean'], 0.05)

    # ----- holds -------------------------------------------------------
    hold_mask = chart.types == TYPE_HOLD
    f['hold_frac'] = float(hold_mask.mean())
    f['hold_coverage'] = _hold_coverage(chart)
    f['hold_max_concurrent'] = float(_max_concurrent_holds(chart))
    if hold_mask.any():
        lengths = chart.hold_ends[hold_mask] - times[hold_mask]
        f['hold_len_mean'] = float(lengths.mean())
        f['hold_len_p90'] = float(np.quantile(lengths, 0.90))
        f['hold_len_max'] = float(lengths.max())
    else:
        f['hold_len_mean'] = f['hold_len_p90'] = f['hold_len_max'] = 0.0

    # ----- finale (尾杀) -------------------------------------------------
    # Official constants weight the hardest *configuration* more than all-song
    # average pressure: a chart can be relaxed for four minutes and still be
    # rated by its last twenty seconds. A chart-wide mean cannot see that, and a
    # 1 s p90 dilutes a 32nd/48th burst. So measure the closing stretch directly.
    tail_start = times[-1] - max(0.25 * duration, 20.0)
    tail_mask = times >= tail_start
    tail_times = press_times[press_times >= tail_start]
    if tail_times.size >= 8:
        span = max(times[-1] - tail_start, 1.0)
        for half in (0.25, 0.5, 1.0):
            curve = _density(tail_times, times[-1], half)
            f[f'finale_d{half:g}_max'] = float(curve.max()) if curve.size else 0.0
        order = np.searchsorted(tail_times, tail_times - 0.02, side='left')
        order2 = np.searchsorted(tail_times, tail_times + 0.02, side='right')
        f['finale_chord_max'] = float((order2 - order - 1).max())
        f['finale_density'] = float(tail_times.size / span)
        f['finale_press_nps'] = float(tail_times.size / span)
    else:
        f['finale_d0.25_max'] = f['finale_d0.5_max'] = f['finale_d1_max'] = 0.0
        f['finale_chord_max'] = f['finale_density'] = f['finale_press_nps'] = 0.0

    # ----- hand-locking and jacks --------------------------------------
    # Smooth jack score: consecutive notes at (nearly) the same x in quick
    # succession. Both factors are kernels, so there is no cutoff to tune.
    if close.any():
        same_place = np.exp(-dx[close] / 0.06)
        quick = np.exp(-dt_safe[close] / 0.12)
        f['jackness'] = float((same_place * quick).mean())
        f['jackness_p95'] = float(np.quantile(same_place * quick, 0.95))
    else:
        f['jackness'] = f['jackness_p95'] = 0.0

    # ----- note type mix -----------------------------------------------
    f['frac_tap'] = float((chart.types == TYPE_TAP).mean())
    f['frac_drag'] = float((chart.types == TYPE_DRAG).mean())
    f['frac_flick'] = float((chart.types == TYPE_FLICK).mean())
    f['frac_below'] = float((~chart.above).mean())
    # "Press" is what the fingers actually have to do: a Tap needs a press and so
    # does the *head* of a Hold. A chart made entirely of Holds (Feeling Blue has
    # 820 of them and zero taps) is a continuous press stream, not an empty chart.
    f['press_nps'] = float(((chart.types == TYPE_TAP) | (chart.types == TYPE_HOLD)).sum()
                           / duration)
    f['tap_nps'] = float((chart.types == TYPE_TAP).sum() / duration)
    f['drag_nps'] = float((chart.types == TYPE_DRAG).sum() / duration)
    f['hold_nps'] = float((chart.types == TYPE_HOLD).sum() / duration)
    f['flick_nps'] = float((chart.types == TYPE_FLICK).sum() / duration)
    f['tap_notes'] = float((chart.types == TYPE_TAP).sum())

    # ----- per-hand load over time --------------------------------------
    press_left = left_weight[press_mask] if press_mask.any() else left_weight
    left_curve = _weighted_density(press_times, press_left, duration, 1.0)
    right_curve = _weighted_density(press_times, 1.0 - press_left, duration, 1.0)
    total_curve = left_curve + right_curve
    f['hand_imbalance_p95'] = float(np.quantile(
        np.abs(left_curve - right_curve) / np.maximum(total_curve, 1e-9), 0.95))
    f['hand_left_peak'] = float(left_curve.max())
    f['hand_right_peak'] = float(right_curve.max())

    # ----- rhythm regularity -------------------------------------------
    positive = dt[dt > 1e-6]
    if positive.size > 4:
        ratio = positive / max(np.median(positive), 1e-9)
        f['interval_cv'] = float(positive.std() / max(positive.mean(), 1e-9))
        f['rhythm_regularity'] = float(np.median(np.abs(np.diff(np.log(ratio)))) if ratio.size > 4 else 0.0)
        # how much of the chart sits on a near-uniform grid
        f['grid_concentration'] = float(np.mean(np.abs(ratio - 1.0) < 0.1))
        f['dt_p10'] = float(np.quantile(positive, 0.10))
        f['dt_p25'] = float(np.quantile(positive, 0.25))
        f['dt_median'] = float(np.median(positive))
        # smooth "how often do notes arrive faster than ~80 ms"
        f['fast_frac'] = float(np.exp(-positive / 0.08).mean())
        if positive.size > 2:
            second = chart.xs[2:] - 2.0 * chart.xs[1:-1] + chart.xs[:-2]
            scale_t = dt_safe[:-1]
            f['x_jagged_mean'] = float((np.abs(second) / scale_t).mean())
            f['x_jagged_p95'] = float(np.quantile(np.abs(second) / scale_t, 0.95))
        else:
            f['x_jagged_mean'] = f['x_jagged_p95'] = 0.0
    else:
        f['interval_cv'] = f['rhythm_regularity'] = f['grid_concentration'] = 0.0
        f['dt_p10'] = f['dt_p25'] = f['dt_median'] = f['fast_frac'] = 0.0
        f['x_jagged_mean'] = f['x_jagged_p95'] = 0.0

    return f


FEATURE_VERSION = 'v12'

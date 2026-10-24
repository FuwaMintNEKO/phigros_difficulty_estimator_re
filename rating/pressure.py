"""Sustained Tap/Hold pressure above a chart's own mean press rate.

Use a full one-second centred box window (half-width 0.5 seconds).  Integrate
its piecewise-constant density exactly on [0, duration]; no sampling grid and no
hand-picked difficulty threshold are involved.  A Hold contributes its head
once.  Drag, Flick, and fake notes never contribute.

pressure_load = mean_density_above_mean * seconds_above_mean / duration
              = integral(density(t) * 1[density(t) > press_count/duration]) / duration

Normalising by chart duration preserves the unit presses/second.  Dividing by
total press count instead would discard absolute intensity and leave a share.
The diagnostic seconds and conditional mean are not separate model features.
"""

from __future__ import annotations

import numpy as np

from .charts import TYPE_HOLD, TYPE_TAP


PRESSURE_WINDOW_SECONDS = 1.0


def pressure_summary(chart) -> dict[str, float]:
    duration = float(chart.duration)
    types = np.asarray(chart.types)
    real = ~np.asarray(chart.fakes, dtype=bool)
    press = real & ((types == TYPE_TAP) | (types == TYPE_HOLD))
    times = np.asarray(chart.times, dtype=float)[press]
    result = {
        'press_count': float(times.size),
        'mean_press_nps': 0.0,
        'high_pressure_seconds': 0.0,
        'high_pressure_mean_nps': 0.0,
        'high_pressure_fraction': 0.0,
        'high_pressure_longest_seconds': 0.0,
        'high_pressure_note_share': 0.0,
        'pressure_load': 0.0,
    }
    if not np.isfinite(times).all() or not np.isfinite(duration):
        raise ValueError('Pressure extraction requires finite timestamps and duration')
    if duration <= 0.0 or times.size == 0:
        return result

    width = PRESSURE_WINDOW_SECONDS
    half = width / 2.0
    enters = np.clip(times - half, 0.0, duration)
    exits = np.clip(times + half, 0.0, duration)
    edges = np.concatenate(([0.0, duration], enters, exits))
    changes = np.concatenate((np.zeros(2, dtype=np.int64),
                              np.ones(times.size, dtype=np.int64),
                              -np.ones(times.size, dtype=np.int64)))
    order = np.argsort(edges, kind='stable')
    edges = edges[order]
    changes = changes[order]
    boundaries, first = np.unique(edges, return_index=True)
    counts = np.cumsum(np.add.reduceat(changes, first))[:-1]
    seconds = np.diff(boundaries)
    density = counts.astype(float) / width
    mean = float(times.size / duration)
    high = density > mean
    high_seconds = float(seconds[high].sum())
    area = float(np.dot(seconds[high], density[high]))

    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    stops = np.flatnonzero(high & ~np.r_[high[1:], False]) + 1
    longest = float(np.max(boundaries[stops] - boundaries[starts])) if starts.size else 0.0
    result.update({
        'mean_press_nps': mean,
        'high_pressure_seconds': high_seconds,
        'high_pressure_mean_nps': area / high_seconds if high_seconds else 0.0,
        'high_pressure_fraction': high_seconds / duration,
        'high_pressure_longest_seconds': longest,
        'high_pressure_note_share': area / times.size,
        'pressure_load': area / duration,
    })
    return result

"""Player-facing difficulty composition.

The model has 106 features with names like ``d1_p90``; nobody wants to read those.
They are grouped into five things a player actually talks about, and the share of
each group is measured by **occlusion**: replace that group's features with the
training median ("an unremarkable chart"), re-predict, and see how much the score
moves. The contributions are additive by construction and share one baseline, so
they can be normalised into a percentage breakdown that adds up to what the model
thinks the chart is worth.
"""

from __future__ import annotations

import numpy as np

CATEGORIES: dict[str, dict] = {
    'volume': {
        'label': '物量密度',
        'hint': '这张谱一共有多少东西要处理',
        'features': ('n_notes', 'duration', 'nps', 'nps_active', 'notes_per_minute',
                     'd1_mean', 'press_nps', 'n_press', 'frac_tap', 'frac_below'),
    },
    'burst': {
        'label': '爆发与尾杀',
        'hint': '最难的那一小段有多极端',
        'features': ('d0.25_max', 'd0.25_p90', 'd0.25_p99', 'd0.25_peak_ratio',
                     'd0.5_max', 'd0.5_p90', 'd0.5_p99', 'd0.5_peak_ratio',
                     'd1_max', 'd1_p90', 'd1_p99', 'd1_peak_ratio',
                     'burst_1s_over_mean', 'peak_sharpness', 'burst_count_per_min'),
    },
    'chords': {
        'label': '多押与双手',
        'hint': '同时要按几个键、左右手怎么分',
        'features': ('ch0.02_mean', 'ch0.02_p95', 'ch0.02_max', 'ch0.02_solo',
                     'spread_50ms_mean', 'spread_50ms_p95', 'spread_150ms_mean',
                     'hand_left_frac', 'hand_balance', 'hand_imbalance_p95',
                     'hand_left_peak', 'hand_right_peak', 'hold_combo',
                     'line_count', 'off_main_line_frac', 'line_entropy', 'line_cross_rate',
                     'ml0.1_notes', 'ml0.1_note_frac', 'ml0.1_lines', 'ml0.1_seconds',
                     'ml0.1_frac', 'ml0.1_density',
                     'ml0.2_notes', 'ml0.2_note_frac', 'ml0.2_lines', 'ml0.2_seconds',
                     'ml0.2_frac', 'ml0.2_density',
                     'ml_weighted_frac', 'ml_weighted_density', 'ch0.02_entropy',
                     'jackness', 'jackness_p95'),
    },
    'movement': {
        'label': '位移与读谱',
        'hint': '眼睛和手要跟多快',
        'features': ('speed_mean', 'speed_median', 'speed_p90', 'speed_log_p90',
                     'speed_p90_over_mean', 'notes_per_height', 'jump_mean', 'jump_close_mean',
                     'jump_speed_mean', 'jump_speed_p95', 'x_jagged_mean',
                     'x_jagged_p95', 'type_switch_load', 'type_switch_flick_rate',
                     'type_switch_drag_rate', 'throughput', 'throughput_p90',
                     'hand_travel_rate', 'flash_load', 'flash_hold_load'),
    },
    'stamina': {
        'label': '体力与节奏',
        'hint': '全程能不能扛住、配置刁不刁',
        'features': ('d2_max', 'd2_p90', 'd2_p99', 'd2_peak_ratio',
                     'd4_max', 'd4_p90', 'd4_p99', 'd4_peak_ratio',
                     'd8_max', 'd8_p90', 'd8_p99', 'd8_peak_ratio',
                     'sustain_4s_over_1s', 'trend_late_early', 'trend_slope',
                     'hold_frac', 'hold_coverage', 'hold_len_mean', 'hold_len_p90',
                     'hold_len_max', 'interval_cv', 'rhythm_regularity',
                     'grid_concentration', 'dt_p10', 'dt_p25', 'dt_median',
                     'fast_frac', 'alternation', 'pressure_load', 'rest_ratio',
                     'density_entropy', 'sustain_frac', 'sustain_flatness', 'sustain_floor'),
    },
}

def missing(columns: list[str]) -> list[str]:
    known = {name for spec in CATEGORIES.values() for name in spec['features']}
    return [name for name in columns if name not in known]


def decomposition(predict, features: dict, baseline: dict, columns: list[str]) -> dict:
    """Per-category contribution in difficulty units (occlusion against baseline)."""
    vector = np.asarray([[features.get(name, 0.0) for name in columns]])

    def scalar(x) -> float:
        out = predict(x)
        return float(out[0]) if hasattr(out, '__len__') else float(out)

    score = scalar(vector)
    index = {name: i for i, name in enumerate(columns)}
    shares = {}
    for key, spec in CATEGORIES.items():
        probe = vector.copy()
        for name in spec['features']:
            if name in index:
                probe[0, index[name]] = baseline[name]
        shares[key] = float(score - scalar(probe))
    total = sum(abs(value) for value in shares.values()) or 1.0
    return {
        'score': score,
        'contributions': shares,
        'percent': {k: 100.0 * v / total for k, v in shares.items()},
    }

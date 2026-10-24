"""Correctness tests for chart parsing, feature extraction and the scale map.

The important one is :func:`test_features_are_format_invariant`: the same chart
written as official JSON and as RPE JSON must produce the same feature vector.
If it does not, the model is secretly learning the file format instead of the
chart, which is exactly the failure mode that mixing official and community
charts invites.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pgr2rpe import load_official, write_rpe
from rating import scale
from rating.charts import load_chart
from rating.features import extract
from test_equivalence import _chart_document


def _synthetic_chart(n_notes: int = 320) -> dict:
    """A deterministic official chart with every note type and some chords.

    The tests must run on a fresh clone, so nothing here reads real charts.
    """
    document = _chart_document(3)
    line = document['judgeLineList'][0]
    notes = []
    for i in range(n_notes):
        kind = (1, 1, 2, 3, 1, 4)[i % 6]
        tick = 40 * i + (7 if i % 5 == 0 else 0)
        notes.append({'type': kind, 'time': tick, 'positionX': ((i * 37) % 17 - 8) * 0.9,
                      'holdTime': 24 if kind == 3 else 0, 'speed': 1.0,
                      'floorPosition': tick * 2.0 * 60.0 / (32 * 180.0)})
        if i % 9 == 0:  # a chord: second note on the same tick
            notes.append({'type': 1, 'time': tick, 'positionX': -((i * 37) % 17 - 8) * 0.9,
                          'holdTime': 0, 'speed': 1.0,
                          'floorPosition': tick * 2.0 * 60.0 / (32 * 180.0)})
    line['notesAbove'] = notes
    line['notesBelow'] = []
    return document


def _write(document: dict) -> str:
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.json', delete=False)
    json.dump(document, handle, ensure_ascii=False)
    handle.close()
    return handle.name


def test_note_count_matches_source():
    document = _synthetic_chart()
    expected = sum(len(line.get('notesAbove', [])) + len(line.get('notesBelow', []))
                   for line in document['judgeLineList'])
    path = _write(document)
    try:
        chart = load_chart(path)
    finally:
        os.unlink(path)
    assert chart.n_notes == expected, (chart.n_notes, expected)
    assert chart.times.size and np.all(np.diff(chart.times) >= 0)


def test_first_note_time_matches_raw_ticks():
    document = _synthetic_chart()
    line = document['judgeLineList'][0]
    tick = min(float(n['time']) for n in line['notesAbove'])
    expected = tick * 60.0 / (32.0 * float(line['bpm']))
    path = _write(document)
    try:
        chart = load_chart(path)
    finally:
        os.unlink(path)
    assert abs(float(chart.times.min()) - expected) < 1e-6, (chart.times.min(), expected)


def test_features_are_finite_and_ordered():
    path = _write(_synthetic_chart())
    try:
        features = extract(load_chart(path))
    finally:
        os.unlink(path)
    assert features
    for name, value in features.items():
        assert math.isfinite(value), (name, value)
    assert features['d0.5_max'] >= features['d1_max'] - 1e-9
    assert features['press_nps'] <= features['nps'] + 1e-9


def test_features_are_format_invariant():
    """official JSON vs the same chart re-encoded as RPE must agree."""
    document = _synthetic_chart()
    official_path = _write(document)
    payload = write_rpe(load_official(json.dumps(document)))
    rpe_path = _write(payload)
    try:
        a = extract(load_chart(official_path))
        b = extract(load_chart(rpe_path))
    finally:
        os.unlink(official_path)
        os.unlink(rpe_path)
    for name, value in a.items():
        tolerance = 1e-6 * max(1.0, abs(value))
        assert abs(value - b[name]) <= tolerance, (name, value, b[name])


def test_zero_length_hold_is_read_as_tap():
    document = _synthetic_chart(60)
    for note in document['judgeLineList'][0]['notesAbove']:
        if note['type'] == 3:
            note['holdTime'] = 0
    path = _write(document)
    try:
        chart = load_chart(path)
    finally:
        os.unlink(path)
    assert int((chart.types == 3).sum()) == 0 and chart.n_notes > 0


def test_sentinel_note_does_not_stretch_duration():
    document = _synthetic_chart()
    document['judgeLineList'][0]['notesAbove'].append(
        {'type': 1, 'time': 10 ** 9, 'positionX': 0.0, 'holdTime': 0, 'speed': 1.0,
         'floorPosition': 0.0})
    path = _write(document)
    try:
        chart = load_chart(path)
    finally:
        os.unlink(path)
    assert chart.duration < 600, chart.duration


def test_scale_hits_the_documented_anchors():
    assert abs(float(scale.official_to_community(17.6)) - 17.6) < 1e-9
    assert abs(float(scale.official_to_community(17.9)) - 18.2) < 1e-9
    assert abs(float(scale.official_to_community(18.0)) - 18.6) < 1e-9
    assert abs(float(scale.community_to_official(17.6)) - 17.6) < 1e-9
    assert abs(float(scale.community_to_official(18.2)) - 17.9) < 1e-9
    assert abs(float(scale.community_to_official(18.6)) - 18.0) < 1e-9
    xs = np.linspace(10, 20, 500)
    mapped = scale.official_to_community(xs)
    assert np.all(np.diff(mapped) > 0), 'map must be strictly increasing'


def test_scale_is_smooth():
    xs = np.linspace(15.0, 18.0, 3001)
    mapped = scale.official_to_community(xs)
    slopes = np.diff(mapped) / np.diff(xs)
    assert slopes.max() < 6.0, slopes.max()
    assert np.max(np.abs(np.diff(slopes))) < 0.05, 'map must not kink'


def test_pressure_matches_dense_reference_and_excludes_contact_notes():
    from dataclasses import replace
    from rating.charts import ChartData, TYPE_TAP, TYPE_HOLD, TYPE_FLICK, TYPE_DRAG
    from rating.pressure import pressure_summary
    times = np.array([0.75, 1.25, 1.25, 2.0, 3.5, 4.25])
    types = np.array([TYPE_TAP, TYPE_HOLD, TYPE_FLICK, TYPE_DRAG, TYPE_TAP, TYPE_HOLD])
    fake = np.array([False, False, False, False, False, True])
    chart = ChartData('test', 'rpe', times, np.zeros(6), types,
                      np.ones(6, dtype=bool), fake, times + 0.1, np.ones(6), 5.0)
    got = pressure_summary(chart)
    expected_times = np.array([0.75, 1.25, 3.5])
    grid = (np.arange(100000) + 0.5) * 5.0 / 100000
    density = (np.abs(grid[:, None] - expected_times[None, :]) < 0.5).sum(axis=1)
    high = density > 3.0 / 5.0
    assert got['press_count'] == 3
    assert abs(got['pressure_load'] - np.mean(density * high)) < 1e-5
    assert abs(got['pressure_load'] - got['high_pressure_mean_nps']
               * got['high_pressure_seconds'] / chart.duration) < 1e-12
    no_press = replace(chart, types=np.full(6, TYPE_DRAG))
    assert pressure_summary(no_press)['pressure_load'] == 0.0
    features = extract(no_press)
    assert all(math.isfinite(v) for v in features.values())
    for key in ('nps', 'press_nps', 'pressure_load'):
        assert features[key] == 0.0, (key, features[key])
    # 零按键谱只返回规模字段，模型侧按 0 补齐
    assert 'd0.25_max' not in features


def test_pressure_duration_normalisation_and_boundary_mass():
    from dataclasses import replace
    from rating.charts import ChartData, TYPE_TAP
    from rating.pressure import pressure_summary
    times = np.array([0.0, 0.5, 1.0, 1.5])
    chart = ChartData('test', 'rpe', times, np.zeros(4), np.full(4, TYPE_TAP),
                      np.ones(4, dtype=bool), np.zeros(4, dtype=bool), times,
                      np.ones(4), 4.0)
    got = pressure_summary(chart)
    assert got['pressure_load'] <= got['mean_press_nps'] + 1e-12
    assert got['high_pressure_seconds'] <= chart.duration
    assert got['high_pressure_note_share'] <= 1.0 + 1e-12
    doubled = replace(chart, times=np.r_[times, times + 4.0], xs=np.zeros(8),
                      types=np.full(8, TYPE_TAP), above=np.ones(8, dtype=bool),
                      fakes=np.zeros(8, dtype=bool), hold_ends=np.r_[times, times + 4.0],
                      speeds=np.ones(8), duration=8.0)
    # Explicit formula identity holds regardless of how repetition changes edges.
    other = pressure_summary(doubled)
    assert abs(other['pressure_load'] - other['high_pressure_mean_nps']
               * other['high_pressure_fraction']) < 1e-12


def _main() -> int:
    failures = 0
    for name, value in sorted(globals().items()):
        if not name.startswith('test_') or not callable(value):
            continue
        try:
            value()
        except AssertionError as exc:
            failures += 1
            print(f'FAIL {name}: {exc}')
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f'ERROR {name}: {type(exc).__name__}: {exc}')
        else:
            print(f'ok   {name}')
    print(f'\n{failures} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())

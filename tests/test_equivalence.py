"""Self-contained equivalence tests for the official <-> RPE converter.

Run with ``python tests/test_equivalence.py`` (no third-party dependencies) or
with pytest, which will collect the ``test_*`` functions.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgr2rpe.model import Chart, Line, Note, Seg, Track
from pgr2rpe.official_in import load_official
from pgr2rpe.official_out import write_official
from pgr2rpe.rpe_in import load_rpe
from pgr2rpe.rpe_out import write_rpe
from pgr2rpe.spec import FormatError, beat_to_seconds, seconds_to_beat, to_triple, from_triple
from pgr2rpe.verify import compare_charts, ConversionError


def _chart_document(format_version: int = 3) -> dict:
    """A small chart that exercises every feature the converter has to preserve."""
    if format_version == 1:
        move = [
            {'startTime': -999999, 'endTime': 0, 'start': 440260, 'end': 440260},
            {'startTime': 0, 'endTime': 1600, 'start': 176156, 'end': 704364},
            {'startTime': 1600, 'endTime': 1000000000, 'start': 704364, 'end': 704364},
        ]
    else:
        move = [
            {'startTime': -999999, 'endTime': 0,
             'start': 0.5, 'end': 0.5, 'start2': 0.5, 'end2': 0.5},
            {'startTime': 0, 'endTime': 1600,
             'start': 0.2, 'end': 0.8, 'start2': 0.3, 'end2': 0.7},
            {'startTime': 1600, 'endTime': 1000000000,
             'start': 0.8, 'end': 0.8, 'start2': 0.7, 'end2': 0.7},
        ]
    return {
        'formatVersion': format_version,
        'offset': 0.125,
        'judgeLineList': [
            {
                'bpm': 180.0,
                'speedEvents': [
                    {'startTime': 0, 'endTime': 3200, 'value': 2.0},
                    {'startTime': 3200, 'endTime': 1000000000, 'value': 1.0},
                ],
                'judgeLineMoveEvents': move,
                'judgeLineRotateEvents': [
                    {'startTime': -999999, 'endTime': 0, 'start': 0.0, 'end': 0.0},
                    {'startTime': 0, 'endTime': 1600, 'start': 0.0, 'end': 120.0},
                    {'startTime': 1600, 'endTime': 1000000000, 'start': 120.0, 'end': 120.0},
                ],
                'judgeLineDisappearEvents': [
                    {'startTime': -999999, 'endTime': 0, 'start': 1.0, 'end': 1.0},
                    {'startTime': 0, 'endTime': 1600, 'start': 1.0, 'end': -0.5},
                    {'startTime': 1600, 'endTime': 1000000000, 'start': 1.4, 'end': 1.4},
                ],
                'notesAbove': [
                    {'type': 1, 'time': 100, 'positionX': 1.5, 'holdTime': 0,
                     'speed': 1.0, 'floorPosition': 2.0 * 100 * 60.0 / (32 * 180.0)},
                    {'type': 3, 'time': 300, 'positionX': -4.25, 'holdTime': 64,
                     'speed': 2.0, 'floorPosition': 2.0 * 300 * 60.0 / (32 * 180.0)},
                    {'type': 4, 'time': 500, 'positionX': 0.0, 'holdTime': 0,
                     'speed': 1.0, 'floorPosition': 2.0 * 500 * 60.0 / (32 * 180.0)},
                ],
                'notesBelow': [
                    {'type': 2, 'time': 200, 'positionX': -1.0, 'holdTime': 0,
                     'speed': 0.5, 'floorPosition': 2.0 * 200 * 60.0 / (32 * 180.0)},
                ],
            },
            {
                'bpm': 90.0,
                'speedEvents': [{'startTime': 0, 'endTime': 1000000000, 'value': 1.5}],
                'judgeLineMoveEvents': [
                    ({'startTime': -999999, 'endTime': 1000000000,
                      'start': 440260, 'end': 440260} if format_version == 1 else
                     {'startTime': -999999, 'endTime': 1000000000,
                      'start': 0.5, 'end': 0.5, 'start2': 0.5, 'end2': 0.5}),
                ],
                'judgeLineRotateEvents': [
                    {'startTime': -999999, 'endTime': 1000000000, 'start': 0.0, 'end': 0.0},
                ],
                'judgeLineDisappearEvents': [
                    {'startTime': -999999, 'endTime': 1000000000, 'start': 1.0, 'end': 1.0},
                ],
                'notesAbove': [
                    {'type': 1, 'time': 64, 'positionX': 2.0, 'holdTime': 0,
                     'speed': 1.0, 'floorPosition': 1.5 * 64 * 60.0 / (32 * 90.0)},
                ],
                'notesBelow': [],
            },
        ],
    }


def _roundtrip(document: dict, **kwargs):
    source = load_official(document)
    payload = json.loads(json.dumps(write_rpe(source, **kwargs), ensure_ascii=False))
    back = load_rpe(payload)
    return source, payload, back


def test_triple_preserves_sign_and_value():
    for beat in (-3.25, -0.5, 0.0, 0.5, 12.125, 1e4 + 0.0625):
        triple = to_triple(beat)
        assert triple[1] >= 0 and triple[2] > 0, triple
        assert abs(from_triple(triple) - beat) < 1e-9, (beat, triple)


def test_track_clipping_preserves_values():
    track = Track([Seg(-1000.0, 0.5, 1.0, 3.0), Seg(0.5, 10.0, 3.0, 3.0)])
    clipped = track.clipped(-5.0, 8.0)
    for i in range(101):
        t = -5.0 + i * 0.13
        assert abs(clipped.at(t) - track.at(t)) < 1e-12, t


def test_track_clamp_splits_at_crossings():
    track = Track([Seg(0.0, 10.0, 0.0, 2.0)])
    clamped = track.clamp_values(0.0, 1.0)
    for i in range(101):
        t = i * 0.1
        assert abs(clamped.at(t) - min(1.0, max(0.0, track.at(t)))) < 1e-12, t


def test_official_to_rpe_is_exact():
    source, payload, back = _roundtrip(_chart_document(3))
    report = compare_charts(source, back)
    assert report.ok, report.issues
    assert report.max_dx < 1e-9 and report.max_dy < 1e-9
    assert report.max_dangle < 1e-6 and report.max_dalpha < 1e-6
    assert report.offset_error < 1e-6


def test_rpe_file_shape():
    _, payload, _ = _roundtrip(_chart_document(3))
    assert payload['META']['RPEVersion'] == 170
    assert payload['META']['offset'] == -125           # official +0.125 s -> music early 125 ms
    assert len(payload['BPMList']) == 1
    assert payload['BPMList'][0]['bpm'] == 180.0       # dominant bpm, the 90 bpm line is scaled
    for line in payload['judgeLineList']:
        assert line['isCover'] == 0, 'back-side notes must stay visible'
        assert line['bpmfactor'] == 1.0
        layer = line['eventLayers'][0]
        assert layer['alphaEvents'], 'alpha must be explicit or the line renders invisible'
        assert all(event['easingType'] == 1 for events in layer.values() for event in events)
        assert all('bezier' in event and 'linkgroup' in event
                   for events in layer.values() for event in events)
    types = sorted(note['type'] for note in payload['judgeLineList'][0]['notes'])
    assert types == [1, 2, 3, 4], types       # official 1/2/3/4 -> RPE 1/4/2/3


def test_type_mapping_roundtrip():
    source, payload, back = _roundtrip(_chart_document(3))
    original = sorted(n.type_id for line in source.lines for n in line.notes)
    restored = sorted(n.type_id for line in back.lines for n in line.notes)
    assert original == restored


def test_hold_end_time_and_length():
    source, payload, back = _roundtrip(_chart_document(3))
    hold_a = next(n for line in source.lines for n in line.notes if n.type_id == 3)
    hold_b = next(n for line in back.lines for n in line.notes if n.type_id == 3)
    assert abs(hold_a.t - hold_b.t) < 1e-9
    assert abs(hold_a.hold - hold_b.hold) < 1e-9
    note = next(n for n in payload['judgeLineList'][0]['notes'] if n['type'] == 2)
    assert note['startTime'] != note['endTime']


def test_format_version_1_is_normalised():
    source_v1, _, back_v1 = _roundtrip(_chart_document(1))
    source_v3, _, back_v3 = _roundtrip(_chart_document(3))
    report = compare_charts(source_v3, back_v1)
    assert report.ok, report.issues
    # v1 packs x/y on an 880x520 grid, so the decoded positions are the same
    # rational numbers as the v3 document only to within one grid cell.
    assert abs(source_v1.lines[0].x.at(0.5) - source_v3.lines[0].x.at(0.5)) < 1e-9


def test_multi_bpm_keeps_absolute_time():
    source, _, back = _roundtrip(_chart_document(3))
    times_a = sorted(round(n.t, 9) for line in source.lines for n in line.notes)
    times_b = sorted(round(n.t, 9) for line in back.lines for n in line.notes)
    assert times_a == times_b
    # line 2 runs at 90 bpm: tick 64 is 64 * 60 / (32 * 90) = 1.3333 s
    assert any(abs(t - 64 * 60 / (32 * 90)) < 1e-9 for t in times_a)


def test_rpe_to_official_roundtrip():
    """The official format stores event times as integer ticks, so this
    direction is exact only to half a tick (0.5 * 60/(32*bpm) seconds)."""
    source, payload, _ = _roundtrip(_chart_document(3))
    rpe_chart = load_rpe(payload)
    official = json.loads(json.dumps(write_official(rpe_chart), ensure_ascii=False))
    again = load_official(official)
    report = compare_charts(source, again, tol=2e-3)
    assert report.max_dtime < 1e-9, report.max_dtime
    assert report.max_dx < 2e-3 and report.max_dy < 2e-3, (report.max_dx, report.max_dy)
    assert report.max_dalpha < 2e-3, report.max_dalpha
    assert report.max_dangle < 0.2, report.max_dangle
    assert report.max_dnote_x < 1e-9 and report.max_dhold < 1e-9


def test_offset_quantisation_is_bounded():
    document = _chart_document(3)
    document['offset'] = 0.1234567
    source = load_official(document)
    payload = write_rpe(source)
    back = load_rpe(json.loads(json.dumps(payload)))
    assert abs(back.offset - source.offset) <= 0.0005 + 1e-12


def test_non_hold_speed_multiplier_is_preserved():
    source, payload, back = _roundtrip(_chart_document(3))
    drag = next(n for line in source.lines for n in line.notes if n.type_id == 2)
    assert abs(drag.speed - 0.5) < 1e-12
    drag_back = next(n for line in back.lines for n in line.notes if n.type_id == 2)
    assert abs(drag_back.speed - 0.5) < 1e-9


def test_rejects_ambiguous_input():
    try:
        load_official({'formatVersion': 3, 'offset': 0,
                       'judgeLineList': [], 'BPMList': []})
    except FormatError:
        pass
    else:
        raise AssertionError('a document with both BPMList and judgeLineList must be rejected')


def test_hold_body_divergence_is_bounded_and_reported():
    """A hold spanning a speed change cannot be exact; the gap must be measured."""
    document = _chart_document(3)
    source = load_official(document)
    _, payload, back = _roundtrip(document)
    report = compare_charts(source, back)
    assert report.ok
    assert report.hold_body_max_error >= 0.0
    # The chart changes speed at tick 3200, outside the hold, so the body is exact.
    assert report.hold_body_max_error < 1e-6


def test_output_satisfies_phira_serde_requirements():
    """Phira deserialises RPE notes/events without serde defaults, so every key
    must be present with the right JSON type or the file will not load at all."""
    _, payload, _ = _roundtrip(_chart_document(3))
    assert isinstance(payload['META']['offset'], int)
    assert isinstance(payload['META']['RPEVersion'], int)
    assert isinstance(payload['BPMList'], list) and payload['BPMList']
    for entry in payload['BPMList']:
        assert isinstance(entry['bpm'], float) and entry['bpm'] > 0
        _assert_triple(entry['startTime'])
    for line in payload['judgeLineList']:
        for key in ('Name', 'Texture', 'isCover', 'eventLayers', 'notes'):
            assert key in line, key
        assert isinstance(line['isCover'], int)
        for layer in line['eventLayers']:
            assert layer is None or isinstance(layer, dict)
            if not layer:
                continue
            for field, events in layer.items():
                assert isinstance(events, list)
                for event in events:
                    for key in ('startTime', 'endTime', 'start', 'end',
                                'easingType', 'easingLeft', 'easingRight', 'linkgroup',
                                'bezier', 'bezierPoints'):
                        assert key in event, (field, key)
                    _assert_triple(event['startTime'])
                    _assert_triple(event['endTime'])
                    assert isinstance(event['easingType'], int)
                    assert isinstance(event['bezierPoints'], list) and len(event['bezierPoints']) == 4
        for note in line['notes']:
            for key in ('type', 'above', 'startTime', 'endTime', 'positionX', 'alpha',
                        'size', 'speed', 'isFake', 'visibleTime', 'yOffset'):
                assert key in note, key
            assert note['type'] in (1, 2, 3, 4)
            assert note['above'] in (0, 1, 2)
            assert isinstance(note['alpha'], int) and 0 <= note['alpha'] <= 255
            assert isinstance(note['isFake'], int)
            _assert_triple(note['startTime'])
            _assert_triple(note['endTime'])


def _assert_triple(value) -> None:
    assert isinstance(value, list) and len(value) == 3, value
    measure, numerator, denominator = value
    assert isinstance(measure, int) and isinstance(numerator, int) and isinstance(denominator, int)
    assert numerator >= 0 and denominator > 0, value


def test_special_effect_fields_are_reported_not_crashing():
    """Chapter 9 charts carry ``blockAreaList`` (the 噪域 / block-area effect)."""
    document = _chart_document(3)
    document['blockAreaList'] = [{
        'topRightPercentage': {'x': 1.5, 'y': 1.2},
        'bottomLeftPercentage': {'x': 0.5, 'y': -0.2},
        'appearTime': 1.0, 'enableTime': 2.0, 'disableTime': 3.0, 'disappearTime': 4.0,
        'isSubtract': False,
        'rotateEvents': [{'anchor': {'x': 1.0, 'y': 0.5}, 'time': 1.0,
                          'easeType': 0, 'rotation': 0.0}],
        'moveEvents': [{'endPosition': {'x': 1.5, 'y': 0.5}, 'time': 1.0,
                        'easeTypeX': 5, 'easeTypeY': 0}],
        'scaleEvents': [],
    }]
    source = load_official(document)
    payload = write_rpe(source)
    back = load_rpe(json.loads(json.dumps(payload)))
    report = compare_charts(source, back)
    assert report.ok, report.issues
    assert source.meta['ignored_fields'] == {'root': ['blockAreaList']}
    assert any('blockAreaList' in warning for warning in source.warnings)
    assert 'blockAreaList' not in payload
    assert 'blockAreaList' not in json.dumps(payload)


def test_unknown_note_field_is_reported():
    document = _chart_document(3)
    document['judgeLineList'][0]['notesAbove'][0]['noiseEvent'] = {'x': 1}
    source = load_official(document)
    assert any('noiseEvent' in warning for warning in source.warnings)


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
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f'ERROR {name}: {type(exc).__name__}: {exc}')
        else:
            print(f'ok   {name}')
    print(f'\n{failures} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())

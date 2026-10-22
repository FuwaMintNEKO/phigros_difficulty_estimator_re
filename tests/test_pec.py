"""Tests for PEC (PE text) -> RPE conversion, checked with the independent oracle.

Run with ``python tests/test_pec.py`` (no dependencies) or with pytest.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgr2rpe import oracle
from pgr2rpe.pec_in import (
    OFFSET_BIAS_MS,
    PEC_SPEED_TO_RPE,
    PecError,
    load_pec,
    pec_to_rpe,
)

DOC = '''175
bp 0.000 180.000
cp 0 0.000 1024.00 700.00
cm 0 0.000 1.000 0.00 0.00 1
cd 0 0.000 0.000
cr 0 0.000 1.000 90.000 5
ca 0 0.000 0
cf 0 0.000 1.000 255
cv 0 0.000 10.000
n1 0 1.000 -512.000 1 0
# 1.50
& 0.80
n2 0 2.000 3.000 0.000 1 0
n3 0 4.000 1024.000 2 0
# 2.00
n4 0 5.000 0.000 1 1
cv 0 2.000 20.000
cp 0 6.000 2048.00 1400.00
'''


def _convert(doc: str = DOC, **kwargs) -> tuple[dict, object]:
    chart = load_pec(doc)
    for note in kwargs.get('notes') or []:
        pass
    return pec_to_rpe(chart, **kwargs), chart


def _write_temp(text: str, suffix: str = '.pec') -> str:
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix=suffix, delete=False)
    handle.write(text)
    handle.close()
    return handle.name


def test_offset_follows_the_documented_175ms_bias():
    payload, chart = _convert()
    assert payload['META']['offset'] == OFFSET_BIAS_MS - 175      # header 175 -> 0
    assert chart.offset_ms == 175
    other = load_pec('0\nbp 0 120\nn1 0 0 1024 1 0\n')
    assert pec_to_rpe(other)['META']['offset'] == 175            # header 0 -> +175 ms


def test_coordinates_use_the_documented_2048x1400_canvas():
    payload, _ = _convert()
    layer = payload['judgeLineList'][0]['eventLayers'][0]
    # First event starts at the centre: PEC (1024, 700) -> (0, 0).
    first = layer['moveXEvents'][0]
    triple = first['startTime']
    assert triple[0] + triple[1] / triple[2] == 0.0 and first['start'] == 0.0
    assert layer['moveYEvents'][0]['start'] == 0.0
    # "cp 0 6 2048 1400" -> x = +675, y = +450.
    assert any(abs(e['end'] - 675.0) < 1e-9 for e in layer['moveXEvents'])
    assert any(abs(e['end'] - 450.0) < 1e-9 for e in layer['moveYEvents'])
    # And "cm ... 0 0 ..." moves to the bottom-left corner: -675, -450.
    assert any(abs(e['end'] + 675.0) < 1e-9 for e in layer['moveXEvents'])
    assert any(abs(e['end'] + 450.0) < 1e-9 for e in layer['moveYEvents'])


def test_note_types_map_one_to_one_and_flags_survive():
    payload, _ = _convert()
    notes = payload['judgeLineList'][0]['notes']
    kinds = sorted(n['type'] for n in notes)
    assert kinds == [1, 2, 3, 4], kinds       # PEC n1..n4 == RPE 1..4
    hold = next(n for n in notes if n['type'] == 2)
    assert hold['startTime'] != hold['endTime']
    flick = next(n for n in notes if n['type'] == 3)
    assert flick['above'] == 0                # PEC "2" == falls from below
    drag = next(n for n in notes if n['type'] == 4)
    assert drag['above'] == 1                 # PEC "1" == falls from above
    assert drag['isFake'] == 1
    tap = next(n for n in notes if n['type'] == 1)
    assert abs(tap['speed'] - 1.5) < 1e-9     # "# 1.50"
    assert abs(tap['size'] - 0.8) < 1e-9      # "& 0.80"
    assert abs(tap['positionX'] + 337.5) < 1e-9   # -512/1024*675


def test_rotation_keeps_its_sign():
    """PEC files and RPE both use clockwise-positive angles."""
    payload, _ = _convert()
    rotate = payload['judgeLineList'][0]['eventLayers'][0]['rotateEvents']
    assert any(abs(e['end'] - 90.0) < 1e-9 for e in rotate), rotate
    assert all(e['end'] >= 0 for e in rotate)


def test_easing_ids_pass_through_and_cf_is_linear():
    payload, _ = _convert()
    layer = payload['judgeLineList'][0]['eventLayers'][0]
    assert any(e['easingType'] == 5 for e in layer['rotateEvents'])
    assert all(e['easingType'] == 1 for e in layer['alphaEvents'])   # cf has no easing


def test_speed_uses_one_documented_scale():
    payload, _ = _convert()
    speed = payload['judgeLineList'][0]['eventLayers'][0]['speedEvents']
    values = [round(e['start'], 6) for e in speed]
    assert abs(values[0] - 10.0 * PEC_SPEED_TO_RPE) < 1e-6
    assert abs(values[-1] - 20.0 * PEC_SPEED_TO_RPE) < 1e-6
    assert all(abs(e['start'] - e['end']) < 1e-12 for e in speed), 'speed must be constant'


def test_conversion_is_exact_against_the_oracle():
    payload, _ = _convert()
    with tempfile.TemporaryDirectory() as folder:
        src = os.path.join(folder, 'chart.pec')
        out = os.path.join(folder, 'chart.rpe.json')
        with open(src, 'w', encoding='utf-8') as handle:
            handle.write(DOC)
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False)
        result = oracle.compare(src, out, samples=400)
    assert result['ok'], result['issues']
    assert max(result['max'].values()) < 1e-9


def test_pec_cannot_become_official():
    from pgr2rpe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(['convert', 'x.json', '-o', 'y.json', '--to', 'official'])
    from pgr2rpe.cli import cmd_convert
    path = _write_temp(DOC)
    try:
        args.input = path
        args.output = os.path.join(tempfile.gettempdir(), 'should_not_exist.json')
        try:
            cmd_convert(args)
        except Exception as exc:      # FormatError is a ValueError
            assert 'official' in str(exc)
        else:
            raise AssertionError('PEC -> official must be rejected')
    finally:
        os.unlink(path)


def test_parser_tolerates_comment_lines_and_unknown_commands():
    chart = load_pec('0\nbp 0 180\n# name: something\nn1 0 1 0 1 0\nzzz 1 2 3\n')
    assert len(chart.lines[0].notes) == 1
    assert any('unknown command' in w for w in chart.warnings)


def test_negative_header_and_held_notes():
    chart = load_pec('-595\nbp 0 120\nn2 0 1 2 -1024 1 0\n')
    payload = pec_to_rpe(chart)
    assert payload['META']['offset'] == 175 + 595
    note = payload['judgeLineList'][0]['notes'][0]
    start = note['startTime'][0] + note['startTime'][1] / note['startTime'][2]
    end = note['endTime'][0] + note['endTime'][1] / note['endTime'][2]
    assert end - start == 1.0


def test_default_speed_is_zero_until_first_cv():
    """Phira inserts (0, 0) when the first cv is after beat 0."""
    chart = load_pec('0\nbp 0 120\ncv 0 4 10\nn1 0 6 0 1 0\n')
    speed = pec_to_rpe(chart)['judgeLineList'][0]['eventLayers'][0]['speedEvents']
    assert speed[0]['start'] == 0.0 and speed[0]['end'] == 0.0
    assert speed[1]['startTime'] != speed[1]['endTime']


def test_duplicate_cv_on_one_beat_keeps_the_last_and_stays_contiguous():
    chart = load_pec('0\nbp 0 120\ncv 0 0 10\ncv 0 2 20\ncv 0 2 30\nn1 0 4 0 1 0\n')
    speed = pec_to_rpe(chart)['judgeLineList'][0]['eventLayers'][0]['speedEvents']
    assert len(speed) == 2, speed          # the redundant cv at beat 2 collapses
    assert abs(speed[1]['start'] - 30 * PEC_SPEED_TO_RPE) < 1e-6


def test_backwards_event_is_dropped_with_a_warning():
    chart = load_pec('0\nbp 0 120\ncr 0 5 3 90 1\nn1 0 6 0 1 0\n')
    payload = pec_to_rpe(chart)
    assert any('ends before it starts' in w for w in chart.warnings)
    # The malformed event must not leak into the output as a backwards range.
    for line in payload['judgeLineList']:
        for events in line['eventLayers'][0].values():
            for event in events:
                start = event['startTime']
                end = event['endTime']
                assert (end[0] + end[1] / end[2]) >= (start[0] + start[1] / start[2])


def test_output_events_are_contiguous_and_non_degenerate():
    payload, _ = _convert()
    def beats(triple):
        return triple[0] + triple[1] / triple[2]
    for line in payload['judgeLineList']:
        for field, events in line['eventLayers'][0].items():
            previous = None
            for event in events:
                start, end = beats(event['startTime']), beats(event['endTime'])
                assert end >= start - 1e-9, (field, event)
                assert abs(end - start) > 1e-12, (field, 'zero-length', event)
                if previous is not None:
                    assert abs(start - previous) < 1e-9, (field, 'gap', previous, start)
                previous = end


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

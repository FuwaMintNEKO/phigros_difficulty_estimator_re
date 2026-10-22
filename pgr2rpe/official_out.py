"""Canonical model -> official Phigros chart JSON.

Used for the round-trip proof (official -> RPE -> model -> official) and to
normalise ``formatVersion 1`` charts into ``formatVersion 3``: the v1
``1000 * x + y`` packing is decoded by :mod:`pgr2rpe.official_in` and re-emitted
as the modern pair of floats.

``formatVersion 1`` output is supported for completeness but is integer only
(the packed field is an integer), so it is quantised to the 880x520 grid; any
position that moves is reported in ``Chart.warnings``.
"""

from __future__ import annotations

import json
from typing import Any

from .model import Chart, Line, Note, Seg, Track
from .spec import (
    FormatError,
    OFFICIAL_X_UNIT,
    TICKS_PER_BEAT,
    seconds_to_tick,
)

_V1_GRID = (880.0, 520.0)
_SENTINEL_START_TICK = -999999.0
_SENTINEL_END_TICK = 1e9


def _round_tick(value: float, warnings: list[str], where: str) -> float:
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        warnings.append(f'{where}: tick {value:.6f} is not an integer, rounded')
    return float(rounded)


def _output_segments(track: Track, line: Line, window: tuple[float, float]) -> list[Seg]:
    """Clip to the window, fill gaps, and extend to the sentinel ticks.

    Official charts are only well formed when the event list is contiguous from
    a very negative tick to a very large one (docs.lchzh.net "相关计算"). Gaps are
    filled with a constant segment, which preserves the clamped value of both
    encodings; the sentinel padding is likewise constant.
    """
    clipped = track.clipped(*window)
    segs = list(clipped.segments())
    if not segs:
        value = track.default
        segs = [Seg(window[0], window[1], value, value)]

    filled: list[Seg] = []
    for seg in segs:
        if filled and seg.t0 > filled[-1].t1 + 1e-12:
            previous = filled[-1]
            filled.append(Seg(previous.t1, seg.t0, previous.v1, previous.v1))
        filled.append(seg)

    sentinel_start = _SENTINEL_START_TICK * 60.0 / (TICKS_PER_BEAT * line.bpm)
    sentinel_end = _SENTINEL_END_TICK * 60.0 / (TICKS_PER_BEAT * line.bpm)
    if filled[0].t0 > sentinel_start:
        filled.insert(0, Seg(sentinel_start, filled[0].t0, filled[0].v0, filled[0].v0))
    if filled[-1].t1 < sentinel_end:
        filled.append(Seg(filled[-1].t1, sentinel_end, filled[-1].v1, filled[-1].v1))
    return filled


def _snap_ticks(segs: list[Seg], line: Line, warnings: list[str], where: str) -> list[float]:
    """Snap every segment boundary to an integer tick, keeping the list ordered.

    Official event times are integers. Rounding boundaries independently could
    make an event end before it starts (or leave a one-tick hole between two
    events), so the shared boundary list is snapped once and reused.
    """
    ticks = [_round_tick(seconds_to_tick(seg.t0, line.bpm), warnings, where) for seg in segs]
    ticks.append(_round_tick(seconds_to_tick(segs[-1].t1, line.bpm), warnings, where))
    for index in range(1, len(ticks)):
        if ticks[index] < ticks[index - 1]:
            ticks[index] = ticks[index - 1]
    return ticks


def _linear_events(track: Track, line: Line, window, warnings, where: str) -> list[dict]:
    segs = _output_segments(track, line, window)
    ticks = _snap_ticks(segs, line, warnings, where)
    return [
        {'startTime': ticks[index], 'endTime': ticks[index + 1],
         'start': round(seg.v0, 9), 'end': round(seg.v1, 9)}
        for index, seg in enumerate(segs)
    ]


def _pack_v1(x_norm: float, y_norm: float, warnings: list[str]) -> int:
    x = (x_norm / 2.0 + 0.5) * _V1_GRID[0]
    y = (y_norm / 2.0 + 0.5) * _V1_GRID[1]
    xr, yr = round(x), round(y)
    if abs(x - xr) > 1e-6 or abs(y - yr) > 1e-6:
        warnings.append('formatVersion 1 output quantises positions to whole grid cells')
    xr = min(int(_V1_GRID[0]), max(0, int(xr)))
    yr = min(int(_V1_GRID[1]), max(0, int(yr)))
    return xr * 1000 + yr


def _move_events(line: Line, window, version: int, warnings) -> list[dict]:
    x_segs = _output_segments(line.x, line, window)
    y_segs = _output_segments(line.y, line, window)
    ticks = _snap_ticks(x_segs, line, warnings, 'moveEvents')
    events = []
    for index, (xs, ys) in enumerate(zip(x_segs, y_segs)):
        start_time = ticks[index]
        end_time = ticks[index + 1]
        if version == 1:
            events.append({'startTime': start_time, 'endTime': end_time,
                           'start': _pack_v1(xs.v0, ys.v0, warnings),
                           'end': _pack_v1(xs.v1, ys.v1, warnings)})
        else:
            events.append({
                'startTime': start_time, 'endTime': end_time,
                'start': round(xs.v0 / 2.0 + 0.5, 9), 'end': round(xs.v1 / 2.0 + 0.5, 9),
                'start2': round(ys.v0 / 2.0 + 0.5, 9), 'end2': round(ys.v1 / 2.0 + 0.5, 9),
            })
    return events


def _note_payload(note: Note, line: Line, speed_track: Track, warnings: list[str]) -> dict:
    return {
        'type': note.type_id,
        'time': _round_tick(seconds_to_tick(note.t, line.bpm), warnings, 'note.time'),
        'positionX': round(note.x / (2.0 * OFFICIAL_X_UNIT), 9),
        'holdTime': _round_tick(seconds_to_tick(note.hold, line.bpm), warnings, 'note.holdTime'),
        'speed': round(note.speed, 9),
        # Recomputed against the speed events that are actually written, so the
        # chart satisfies p_N = P_J(t_N) with its own (padded) speed list.
        'floorPosition': round(speed_track.integral(note.t) + note.floor_offset, 9),
    }


def write_official(chart: Chart, *, format_version: int = 3,
                   window: tuple[float, float] | None = None,
                   strip_legacy_fields: bool = True) -> dict[str, Any]:
    if format_version not in (1, 3):
        raise FormatError('official output supports formatVersion 1 or 3')
    warnings: list[str] = []
    window = window or chart.time_window()

    lines = []
    for line in chart.lines:
        payload: dict[str, Any] = {'bpm': round(line.bpm, 9)}
        payload['judgeLineMoveEvents'] = _move_events(line, window, format_version, warnings)
        payload['judgeLineRotateEvents'] = _linear_events(line.angle, line, window, warnings,
                                                          'rotateEvents')
        payload['judgeLineDisappearEvents'] = _linear_events(line.alpha, line, window, warnings,
                                                             'disappearEvents')
        speed_segments = _output_segments(line.speed, line, window)
        speed_track = Track(speed_segments)
        speed_ticks = _snap_ticks(speed_segments, line, warnings, 'speedEvents')
        payload['speedEvents'] = [
            {'startTime': speed_ticks[index], 'endTime': speed_ticks[index + 1],
             'value': round(seg.v0, 9)}
            for index, seg in enumerate(speed_segments)
        ]

        above = [_note_payload(n, line, speed_track, warnings) for n in line.notes if n.above]
        below = [_note_payload(n, line, speed_track, warnings) for n in line.notes if not n.above]
        payload['notesAbove'] = above
        payload['notesBelow'] = below
        if not strip_legacy_fields:
            payload['numOfNotes'] = len(above) + len(below)
            payload['numOfNotesAbove'] = len(above)
            payload['numOfNotesBelow'] = len(below)
        lines.append(payload)

    result: dict[str, Any] = {
        'formatVersion': format_version,
        'offset': round(chart.offset, 9),
        'judgeLineList': lines,
    }
    if not strip_legacy_fields:
        result['numOfNotes'] = sum(len(l['notesAbove']) + len(l['notesBelow']) for l in lines)
    chart.warnings.extend(dict.fromkeys(warnings))
    return result


def write_official_file(chart: Chart, path: str, **kwargs) -> dict:
    payload = write_official(chart, **kwargs)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    return payload

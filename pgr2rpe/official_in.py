"""Official Phigros chart JSON -> canonical model.

Format reference: https://docs.lchzh.net/learning/phigros/
Cross-checked against Phira's ``prpr/src/parse/pgr.rs``.

Anything that the reference toolchain silently tolerated but that changes the
meaning of a chart is reported in ``Chart.warnings`` rather than swallowed.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import Any

from .model import Chart, Line, Note, Seg, Track
from .spec import FormatError, OFFICIAL_X_UNIT, tick_to_seconds

_MOVE_FIRST_GRID = (880.0, 520.0)   # formatVersion 1: bottom-left origin, top-right (880, 520)
_OTHER_UNIT_IN_HEIGHTS = 0.1        # formatVersion not in {1, 3}

_ROOT_KEYS = {'formatVersion', 'offset', 'judgeLineList', 'numOfNotes'}
_LINE_KEYS = {'bpm', 'notesAbove', 'notesBelow', 'notes', 'speedEvents',
              'judgeLineMoveEvents', 'judgeLineRotateEvents', 'judgeLineDisappearEvents',
              'numOfNotes', 'numOfNotesAbove', 'numOfNotesBelow'}
_NOTE_KEYS = {'type', 'time', 'positionX', 'holdTime', 'speed', 'floorPosition'}
_EVENT_KEYS = {'startTime', 'endTime', 'start', 'end', 'start2', 'end2', 'value',
               'floorPosition'}

SPECIAL_FIELDS = {
    'blockAreaList': '噪域/遮罩区 (Chapter 9 block-area effect)',
}
"""Known non-convertible fields.

``blockAreaList`` describes rectangular mask areas with their own
appear/enable/disable/disappear times (in seconds, not ticks) and move/rotate/
scale sub-events. It is a purely visual effect - it never affects judgement -
and Re:PhiEdit has no equivalent, so the converter reports it and drops it.
"""


def _opt(raw: dict, key: str, default: float) -> float:
    """Read a numeric field, treating a JSON null as "not set".

    Some charts carry explicit nulls (``"speed": null``) instead of omitting the
    key; that is the editor writing a default, not a malformed value.
    """
    value = raw.get(key, default)
    return default if value is None else value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FormatError(f'{field} must be a number, got {type(value).__name__}')
    result = float(value)
    if not math.isfinite(result):
        raise FormatError(f'{field} must be finite')
    return result


def _list(container: dict, key: str, path: str, warnings: list[str]) -> list[dict]:
    value = container.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise FormatError(f'{path}.{key} must be a list')
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise FormatError(f'{path}.{key}[{index}] must be an object')
        result.append(item)
    return result


def _event_track(
    events: list[dict],
    *,
    bpm: float,
    path: str,
    pick: callable,
    warnings: list[str],
    constant: bool = False,
) -> Track:
    """Build a track from an official event list (times in ticks)."""
    segs: list[Seg] = []
    previous_start: float | None = None
    for index, event in enumerate(events):
        where = f'{path}[{index}]'
        if 'startTime' not in event or 'endTime' not in event:
            raise FormatError(f'{where} is missing startTime/endTime')
        t0 = tick_to_seconds(_number(event['startTime'], f'{where}.startTime'), bpm)
        t1 = tick_to_seconds(_number(event['endTime'], f'{where}.endTime'), bpm)
        v0 = _number(pick(event, 'start', where), f'{where}.start')
        if t1 < t0:
            warnings.append(f'{where}: endTime < startTime, segment dropped')
            continue
        if constant:
            v1 = v0
        else:
            v1 = _number(pick(event, 'end', where), f'{where}.end')
        if previous_start is not None and t0 < previous_start - 1e-12:
            warnings.append(f'{where}: events are not sorted by startTime')
        previous_start = t0
        segs.append(Seg(t0, t1, v0, v1))
    if not segs:
        return Track.constant(0.0)
    segs.sort(key=lambda s: (s.t0, s.t1))
    overlaps = sum(1 for a, b in zip(segs, segs[1:]) if b.t0 < a.t1 - 1e-9)
    if overlaps:
        warnings.append(f'{path}: {overlaps} overlapping event(s), later event wins')
    return Track(segs)


def _move_track(events: list[dict], *, bpm: float, path: str, version: int,
                axis: int, warnings: list[str]) -> Track:
    """Decode judgeLineMoveEvents for one axis into normalised coordinates.

    formatVersion 1: ``start``/``end`` are ``1000 * x + y`` on a 880x520 grid
                     with the origin at the bottom-left.
    formatVersion 3: ``start``/``end`` are x, ``start2``/``end2`` are y, both in
                     [0, 1] with the origin at the bottom-left.
    anything else:  x and y are centred coordinates in units of 0.1 screen
                     height (assumed 16:9, as in the reference renderers).
    """
    if version == 3:
        key_start, key_end = ('start', 'end') if axis == 0 else ('start2', 'end2')
    else:
        # formatVersion 1 packs both axes into one integer; the undocumented
        # branch uses centred coordinates on both axes.
        key_start, key_end = 'start', 'end'

    def decode(raw: float) -> float:
        if version == 1:
            value = int(round(raw))
            grid = _MOVE_FIRST_GRID[axis]
            if axis == 0:
                picked = math.floor(value / 1000.0)
            else:
                picked = value - math.floor(value / 1000.0) * 1000.0
            return (picked / grid - 0.5) * 2.0
        if version == 3:
            return (raw - 0.5) * 2.0
        # centred, unit = 0.1 H on both axes; assume 16:9
        unit_norm = (2.0 * _OTHER_UNIT_IN_HEIGHTS) if axis == 1 else (
            2.0 * _OTHER_UNIT_IN_HEIGHTS * 9.0 / 16.0)
        return raw * unit_norm

    segs: list[Seg] = []
    for index, event in enumerate(events):
        where = f'{path}[{index}]'
        if 'startTime' not in event or 'endTime' not in event:
            raise FormatError(f'{where} is missing startTime/endTime')
        t0 = tick_to_seconds(_number(event['startTime'], f'{where}.startTime'), bpm)
        t1 = tick_to_seconds(_number(event['endTime'], f'{where}.endTime'), bpm)
        raw0 = _number(event.get(key_start, 0.0), f'{where}.{key_start}')
        raw1 = _number(event.get(key_end, 0.0), f'{where}.{key_end}')
        if t1 < t0:
            warnings.append(f'{where}: endTime < startTime, segment dropped')
            continue
        segs.append(Seg(t0, t1, decode(raw0), decode(raw1)))
    if not segs:
        return Track.constant(0.0)
    segs.sort(key=lambda s: (s.t0, s.t1))
    return Track(segs)


def _collect_unknown(container: dict, known: set[str], level: str,
                     ignored: dict[str, list[str]]) -> None:
    extra = sorted(set(container) - known)
    if extra:
        ignored.setdefault(level, []).extend(extra)


def _read_notes(line: dict, index: int, bpm: float, speed_track: Track,
                warnings: list[str]) -> list[Note]:
    path = f'judgeLineList[{index}]'
    has_above = 'notesAbove' in line
    has_below = 'notesBelow' in line
    has_legacy = 'notes' in line
    if has_legacy and (has_above or has_below):
        raise FormatError(f'{path} mixes notes with notesAbove/notesBelow')

    collections: list[tuple[bool, list[dict]]] = []
    if has_legacy:
        collections.append((True, _list(line, 'notes', path, warnings)))
    else:
        collections.append((True, _list(line, 'notesAbove', path, warnings)))
        collections.append((False, _list(line, 'notesBelow', path, warnings)))

    notes: list[Note] = []
    for above, raw_notes in collections:
        for note_index, raw in enumerate(raw_notes):
            where = f'{path}.{"notesAbove" if above else "notesBelow"}[{note_index}]'
            type_raw = _number(raw.get('type'), f'{where}.type')
            type_id = int(type_raw)
            if type_id != type_raw or type_id not in (1, 2, 3, 4):
                warnings.append(f'{where}: type {type_raw!r} is not 1..4, kept as-is')
                if type_id not in (1, 2, 3, 4):
                    raise FormatError(f'{where}: unsupported note type {type_raw!r}')
            if 'time' not in raw:
                raise FormatError(f'{where}: missing time')
            time_ticks = _number(raw['time'], f'{where}.time')
            hold_ticks = _number(_opt(raw, 'holdTime', 0.0), f'{where}.holdTime')
            if type_id == 3 and hold_ticks <= 0:
                # A Hold with no length behaves exactly like a Tap: press and
                # release in the same instant. Charts do contain these (a typo, or
                # an editor's "hold" left at 0), so degrade it instead of refusing
                # to load the whole chart.
                warnings.append(f'{where}: hold note with holdTime {hold_ticks}, '
                                f'treated as Tap')
                type_id, hold_ticks = 1, 0.0
            if type_id != 3 and hold_ticks != 0:
                warnings.append(f'{where}: non-hold note carries holdTime {hold_ticks}, zeroed')
                hold_ticks = 0.0
            x_units = _number(_opt(raw, 'positionX', 0.0), f'{where}.positionX')
            time_seconds = tick_to_seconds(time_ticks, bpm)
            floor_absolute = _number(_opt(raw, 'floorPosition', 0.0),
                                     f'{where}.floorPosition')
            # floorPosition is an absolute vertical position; store the residual
            # against the line's own speed integral so it is origin independent.
            notes.append(Note(
                type_id=type_id,
                t=time_seconds,
                x=x_units * 2.0 * OFFICIAL_X_UNIT,          # official X -> [-1, 1]
                speed=_number(_opt(raw, 'speed', 1.0), f'{where}.speed'),
                hold=tick_to_seconds(hold_ticks, bpm),
                above=above,
                fake=False,
                floor_offset=floor_absolute - speed_track.integral(time_seconds),
            ))
    return notes


def load_official(data: dict | str | bytes, *, source: str = '') -> Chart:
    """Parse an official chart (JSON text or already-decoded dict)."""
    if isinstance(data, (str, bytes, bytearray)):
        text = data.decode('utf-8-sig') if isinstance(data, (bytes, bytearray)) else data
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f'not valid JSON: {exc}') from exc
    elif isinstance(data, dict):
        payload = data
    else:
        raise FormatError('chart must be JSON text or a dict')

    if not isinstance(payload, dict):
        raise FormatError('chart root must be an object')
    if 'judgeLineList' not in payload:
        raise FormatError('not an official chart: missing judgeLineList')
    if 'BPMList' in payload or 'META' in payload:
        raise FormatError('this looks like an RPE chart, not an official chart')

    warnings: list[str] = []
    version = payload.get('formatVersion', 3)
    if not isinstance(version, int):
        raise FormatError('formatVersion must be an integer')
    if version not in (1, 3):
        warnings.append(f'formatVersion {version} is undocumented, assuming centred 0.1H units')

    offset = _number(payload.get('offset', 0.0), 'offset')
    raw_lines = payload.get('judgeLineList')
    if not isinstance(raw_lines, list) or not raw_lines:
        raise FormatError('judgeLineList must be a non-empty list')

    lines: list[Line] = []
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, dict):
            raise FormatError(f'judgeLineList[{index}] must be an object')
        path = f'judgeLineList[{index}]'
        if 'bpm' not in raw_line:
            raise FormatError(f'{path} is missing bpm')
        bpm = _number(raw_line['bpm'], f'{path}.bpm')
        if bpm <= 0:
            raise FormatError(f'{path}.bpm must be positive')

        move_events = _list(raw_line, 'judgeLineMoveEvents', path, warnings)
        x_track = _move_track(move_events, bpm=bpm, path=f'{path}.judgeLineMoveEvents',
                              version=version, axis=0, warnings=warnings)
        y_track = _move_track(move_events, bpm=bpm, path=f'{path}.judgeLineMoveEvents',
                              version=version, axis=1, warnings=warnings)

        rotate_events = _list(raw_line, 'judgeLineRotateEvents', path, warnings)
        angle_track = _event_track(rotate_events, bpm=bpm, path=f'{path}.judgeLineRotateEvents',
                                   pick=lambda e, k, w: e.get(k, 0.0), warnings=warnings)

        alpha_events = _list(raw_line, 'judgeLineDisappearEvents', path, warnings)
        alpha_track = _event_track(alpha_events, bpm=bpm, path=f'{path}.judgeLineDisappearEvents',
                                   pick=lambda e, k, w: e.get(k, 0.0), warnings=warnings)
        if not alpha_track:
            alpha_track = Track.constant(1.0)
        else:
            low = min(min(s.v0, s.v1) for s in alpha_track.segments())
            high = max(max(s.v0, s.v1) for s in alpha_track.segments())
            if low < 0.0 or high > 1.0:
                warnings.append(f'{path}.judgeLineDisappearEvents: values outside [0, 1] '
                                f'({low:g}..{high:g}) will be clamped')

        speed_events = _list(raw_line, 'speedEvents', path, warnings)
        speed_track = _event_track(speed_events, bpm=bpm, path=f'{path}.speedEvents',
                                   pick=lambda e, k, w: e.get('value', 0.0),
                                   warnings=warnings, constant=True)
        if not speed_track:
            warnings.append(f'{path}: no speedEvents, assuming 1 Y/s')
            speed_track = Track.constant(1.0)

        notes = _read_notes(raw_line, index, bpm, speed_track, warnings)
        for note in notes:
            residual = note.floor_offset
            if abs(residual) > 0.01:
                warnings.append(
                    f'{path}: note at {note.t:.3f}s has floorPosition off the line '
                    f'integral by {residual:.3f} Y (kept as the RPE yOffset)')
                break

        lines.append(Line(bpm=bpm, x=x_track, y=y_track, angle=angle_track,
                          alpha=alpha_track, speed=speed_track, notes=notes))

    ignored: dict[str, list[str]] = {}
    _collect_unknown(payload, _ROOT_KEYS, 'root', ignored)
    for index, raw_line in enumerate(raw_lines):
        _collect_unknown(raw_line, _LINE_KEYS, f'judgeLineList[{index}]', ignored)
        for field in ('speedEvents', 'judgeLineMoveEvents', 'judgeLineRotateEvents',
                      'judgeLineDisappearEvents'):
            for event in raw_line.get(field) or []:
                if isinstance(event, dict):
                    _collect_unknown(event, _EVENT_KEYS, f'{field}', ignored)
        for side in ('notesAbove', 'notesBelow', 'notes'):
            for note in raw_line.get(side) or []:
                if isinstance(note, dict):
                    _collect_unknown(note, _NOTE_KEYS, 'note', ignored)
    for level, keys in ignored.items():
        for key in dict.fromkeys(keys):
            label = SPECIAL_FIELDS.get(key)
            suffix = f' [{label}]' if label else ''
            warnings.append(f'ignored unsupported field {level}.{key}{suffix}: '
                            f'no RPE equivalent, dropped from the output')

    chart = Chart(offset=offset, lines=lines, format_version=version, source=source,
                  meta=_read_meta(payload), warnings=warnings)
    chart.meta['ignored_fields'] = {level: sorted(set(keys)) for level, keys in ignored.items()}
    chart.check()
    return chart


def _read_meta(payload: dict) -> dict:
    meta = {k: payload[k] for k in ('numOfNotes', 'song', 'name') if k in payload}
    meta['official_format_version'] = payload.get('formatVersion')
    return meta

"""Canonical model -> Re:PhiEdit (RPE) JSON.

Design rules, in order of importance:

1. **Time is preserved in seconds.** RPE's beat axis is global, while official
   charts give every judge line its own BPM. Each tick is therefore converted to
   seconds with its own line's BPM and then to a beat on a single global BPM
   axis. ``bpmfactor`` stays 1.0: Re:PhiEdit cannot edit that field and Phira
   ignores it (``// TODO bpmfactor``), so relying on it would make the file
   render incorrectly in the most common RPE-compatible player.
2. **Every observable quantity is mapped through an exact unit ratio**, never a
   fitted constant (see :mod:`pgr2rpe.spec`).
3. **Lines always get an explicit alpha track.** Both Phira and the RPE docs
   treat a missing alpha event list as transparent, while a missing disappear
   list in an official chart means opaque.
"""

from __future__ import annotations

import math
from typing import Any

from .model import Chart, Line, Note, Track
from .spec import (
    DEFAULT_RPE_VERSION,
    OFFICIAL_SPEED_TO_RPE,
    RPE_ALPHA_MAX,
    RPE_ROTATE_SIGN,
    RPE_TYPE_TO_OFFICIAL,
    X_PX_PER_NORM,
    Y_PX_PER_NORM,
    Y_PX_PER_OFFICIAL_UNIT,
    seconds_to_beat,
    to_triple,
)

_OFFICIAL_TO_RPE_TYPE = {v: k for k, v in RPE_TYPE_TO_OFFICIAL.items()}
_EVENT_DEFAULTS = {
    'easingType': 1,      # linear: both official event types interpolate linearly
    'easingLeft': 0.0,
    'easingRight': 1.0,
    'linkgroup': 0,
    'bezier': 0,
    'bezierPoints': [0.0, 0.0, 0.0, 0.0],
}

FLOOR_POLICY_AUTO = 'auto'
FLOOR_POLICY_DROP = 'drop'
FLOOR_POLICY_YOFFSET = 'yoffset'   # accepted as an alias of 'auto' for older callers

_POLICIES = (FLOOR_POLICY_AUTO, FLOOR_POLICY_DROP, FLOOR_POLICY_YOFFSET)


def _round(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result  # normalise -0.0


def _event(t0_beat: float, t1_beat: float, v0: float, v1: float) -> dict[str, Any]:
    event = {
        'startTime': to_triple(t0_beat),
        'endTime': to_triple(t1_beat),
        'start': _round(v0, 9),
        'end': _round(v1, 9),
    }
    event.update(_EVENT_DEFAULTS)
    return event


def _emit_track(track: Track, scale: float, window: tuple[float, float],
                bpm: float) -> list[dict] | None:
    """Convert a canonical track into RPE events (values multiplied by ``scale``)."""
    b0 = seconds_to_beat(window[0], bpm)
    b1 = seconds_to_beat(window[1], bpm)
    if not track.segments():
        # No events at all: RPE's default is 0 (centre / no rotation), so a zero
        # default can be omitted, but a non-zero constant must be written out.
        if abs(track.default) < 1e-12:
            return None
        value = track.default * scale
        return [_event(b0, b1, value, value)]
    clipped = track.clipped(*window)
    if clipped.is_constant(1e-12):
        value = clipped.at(window[0]) * scale
        return [_event(b0, b1, value, value)]
    return [_event(seconds_to_beat(seg.t0, bpm), seconds_to_beat(seg.t1, bpm),
                   seg.v0 * scale, seg.v1 * scale) for seg in clipped.compressed().segments()]


def _alpha_events(line: Line, window: tuple[float, float], bpm: float) -> list[dict]:
    """Official disappear (0..1) -> RPE alpha (0..255), always emitted.

    A judge line without alpha events is drawn transparent by Phira
    (``Anim::now()`` falls back to 0) while an official chart without disappear
    events is fully opaque, so the conversion must synthesise an opaque track.
    """
    track = line.alpha.clamp_values(0.0, 1.0)
    b0 = seconds_to_beat(window[0], bpm)
    b1 = seconds_to_beat(window[1], bpm)

    if not track.segments():
        value = track.default * RPE_ALPHA_MAX
        return [_event(b0, b1, value, value)]
    clipped = track.clipped(*window)
    if clipped.is_constant(1e-12):
        value = clipped.at(window[0]) * RPE_ALPHA_MAX
        return [_event(b0, b1, value, value)]
    return [_event(seconds_to_beat(seg.t0, bpm), seconds_to_beat(seg.t1, bpm),
                   seg.v0 * RPE_ALPHA_MAX, seg.v1 * RPE_ALPHA_MAX)
            for seg in clipped.compressed().segments()]


def _speed_events(line: Line, window: tuple[float, float], bpm: float) -> list[dict]:
    """Official speed (constant per event, Y/s) -> RPE speed (RPESpeed).

    ``start == end`` keeps the segment constant, which is required: an official
    speed event holds one value over its whole interval and jumps at the
    boundary, whereas RPE interpolates between ``start`` and ``end``.
    """
    track = line.speed
    b0 = seconds_to_beat(window[0], bpm)
    b1 = seconds_to_beat(window[1], bpm)
    if not track.segments():
        value = track.default * OFFICIAL_SPEED_TO_RPE
        return [_event(b0, b1, value, value)]
    clipped = track.clipped(*window)
    if clipped.is_constant(1e-12):
        value = clipped.at(window[0]) * OFFICIAL_SPEED_TO_RPE
        return [_event(b0, b1, value, value)]
    return [_event(seconds_to_beat(seg.t0, bpm), seconds_to_beat(seg.t1, bpm),
                   seg.v0 * OFFICIAL_SPEED_TO_RPE, seg.v1 * OFFICIAL_SPEED_TO_RPE)
            for seg in clipped.segments()]


def _note_payload(note: Note, bpm: float) -> dict[str, Any]:
    start_beat = seconds_to_beat(note.t, bpm)
    end_beat = seconds_to_beat(note.t_end, bpm) if note.is_hold else start_beat
    return {
        'type': _OFFICIAL_TO_RPE_TYPE[note.type_id],
        'above': 1 if note.above else 0,
        'startTime': to_triple(start_beat),
        'endTime': to_triple(end_beat),
        'positionX': _round(note.x * X_PX_PER_NORM, 9),
        'alpha': 255,
        'size': 1.0,
        # Official Hold notes carry an absolute tail speed (always equal to the
        # line speed at judgment); RPE holds are integral based, so 1.0 is the
        # exact multiplier. Non-hold official notes use speed as a visual
        # distance multiplier, which is exactly what RPE's field means too.
        'speed': 1.0 if note.is_hold else _round(note.speed, 9),
        'isFake': 1 if note.fake else 0,
        'visibleTime': 999999.0,
        'yOffset': 0.0,
    }


def _floor_offset(note: Note, line: Line, policy: str) -> float:
    """RPE has no floorPosition: the residual becomes the note's yOffset.

    A residual is not a defect: some charts park notes far from their judge line
    on purpose (``speed = 0`` sections, invisible notes judged by time), so the
    default policy writes it verbatim. 1e-9 Y is float noise (~5e-7 px).
    """
    if policy == FLOOR_POLICY_DROP:
        return 0.0
    residual = note.floor_offset
    if abs(residual) <= 1e-9:
        return 0.0
    return residual * Y_PX_PER_OFFICIAL_UNIT


def write_rpe(
    chart: Chart,
    *,
    bpm: float | None = None,
    meta: dict[str, Any] | None = None,
    rpe_version: int = DEFAULT_RPE_VERSION,
    floor_policy: str = FLOOR_POLICY_AUTO,
    window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Serialise a canonical chart as an RPE chart dict."""
    if floor_policy not in _POLICIES:
        raise ValueError(f'unknown floor policy: {floor_policy}')

    warnings: list[str] = []
    global_bpm = float(bpm if bpm is not None else chart.dominant_bpm())
    if global_bpm <= 0 or not math.isfinite(global_bpm):
        raise ValueError('global bpm must be a positive finite number')
    window = window or chart.time_window()

    offset_ms = -float(chart.offset) * 1000.0
    if abs(offset_ms - round(offset_ms)) > 1e-6:
        warnings.append(f'offset {chart.offset!r}s quantised to {round(offset_ms)}ms '
                        f'(RPE stores integer milliseconds)')

    metadata = {
        'name': 'Converted chart',
        'composer': '',
        'charter': '',
        'level': '',
        'song': '',
        'background': 'background.png',
        'illustration': 'illustration.png',
        'id': 0,
    }
    for key, value in (meta or {}).items():
        metadata[key] = value
    metadata['RPEVersion'] = int(rpe_version)
    metadata['offset'] = int(round(offset_ms))

    lines_out: list[dict[str, Any]] = []
    for line_index, line in enumerate(chart.lines):
        move_x = _emit_track(line.x, X_PX_PER_NORM, window, global_bpm)
        move_y = _emit_track(line.y, Y_PX_PER_NORM, window, global_bpm)
        rotate = _emit_track(line.angle, RPE_ROTATE_SIGN, window, global_bpm)
        alpha = _alpha_events(line, window, global_bpm)
        speed = _speed_events(line, window, global_bpm)

        layer: dict[str, Any] = {'alphaEvents': alpha, 'speedEvents': speed}
        if move_x is not None:
            layer['moveXEvents'] = move_x
        if move_y is not None:
            layer['moveYEvents'] = move_y
        if rotate is not None:
            layer['rotateEvents'] = rotate

        notes_out = []
        for note in sorted(line.notes, key=lambda n: (n.t, not n.above, n.x)):
            payload = _note_payload(note, global_bpm)
            payload['yOffset'] = _round(_floor_offset(note, line, floor_policy), 6)
            notes_out.append(payload)

        lines_out.append({
            'Group': 0,
            'Name': f'line{line_index}',
            'Texture': 'line.png',
            'anchor': [0.5, 0.5],
            'posControl': [],
            'sizeControl': [],
            'skewControl': [],
            'yControl': [],
            'alphaControl': [],
            'eventLayers': [layer],
            'extended': {
                'colorEvents': [], 'textEvents': [], 'scaleXEvents': [],
                'scaleYEvents': [], 'inclineEvents': [], 'paintEvents': [], 'gifEvents': [],
            },
            'father': -1,
            'rotateWithFather': False,
            'isCover': 0,   # 1 would hide every note on the back side
            'zOrder': 0,
            'bpmfactor': 1.0,
            'notes': notes_out,
            'numOfNotes': sum(1 for n in line.notes if not n.is_hold),
        })

    result = {
        'BPMList': [{'bpm': _round(global_bpm, 9), 'startTime': [0, 0, 1]}],
        'META': metadata,
        'chartTime': _round(max(window[1], chart.last_note_time), 3),
        'judgeLineGroup': ['A'],
        'judgeLineList': lines_out,
        'multiLineString': '',
        'multiScale': 1.0,
        'xybind': False,
    }
    chart.warnings.extend(dict.fromkeys(warnings))
    return result


def write_rpe_file(chart: Chart, path: str, **kwargs: Any) -> dict[str, Any]:
    import json
    payload = write_rpe(chart, **kwargs)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
    return payload

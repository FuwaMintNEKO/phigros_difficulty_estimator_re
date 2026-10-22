"""Re:PhiEdit (RPE) JSON -> canonical model.

This reader exists mainly so the converter can be *proved* correct: the writer
produces RPE, this module reads it back independently, and
:mod:`pgr2rpe.verify` compares the two canonical charts.

Scope: the subset Re:PhiEdit charts use for note geometry - linear
(``easingType == 1``) events, no bezier. Non-linear easing is rejected with an
explicit message instead of being silently approximated.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .model import Chart, Line, Note, Seg, Track
from .spec import (
    FormatError,
    OFFICIAL_SPEED_TO_RPE,
    RPE_ALPHA_MAX,
    RPE_ROTATE_SIGN,
    RPE_TYPE_TO_OFFICIAL,
    X_PX_PER_NORM,
    Y_PX_PER_NORM,
    Y_PX_PER_OFFICIAL_UNIT,
    from_triple,
)


class BpmAxis:
    """Global beat axis: beat -> seconds, using RPE's piecewise-constant BPM list."""

    def __init__(self, entries: list[dict]):
        points: list[tuple[float, float]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise FormatError(f'BPMList[{index}] must be an object')
            beat = from_triple(entry.get('startTime', [0, 0, 1]))
            bpm = float(entry.get('bpm', 0.0))
            if bpm <= 0:
                raise FormatError(f'BPMList[{index}].bpm must be positive')
            points.append((beat, bpm))
        if not points:
            points = [(0.0, 120.0)]
        points.sort()
        if points[0][0] > 0:
            points.insert(0, (0.0, points[0][1]))
        self.points = points

    @property
    def bpm_at_zero(self) -> float:
        return self.points[0][1]

    def time(self, beat: float) -> float:
        if beat <= self.points[0][0]:
            # Events may live before beat 0 (the writer clips its window at a
            # negative time); integrate backwards with the first BPM.
            return (beat - self.points[0][0]) * 60.0 / self.points[0][1]
        total = 0.0
        for index, (start, bpm) in enumerate(self.points):
            end = self.points[index + 1][0] if index + 1 < len(self.points) else beat
            if beat <= start:
                break
            seg_end = min(beat, end)
            if seg_end > start:
                total += (seg_end - start) * 60.0 / bpm
            if beat <= end:
                break
        return total


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormatError(f'{field} must be a number')
    value = float(value)
    if not math.isfinite(value):
        raise FormatError(f'{field} must be finite')
    return value


def _linear_track(events: list[dict], time_of_beat, scale: float,
                  path: str, warnings: list[str],
                  allow_easing_loss: bool = False) -> Track:
    segs: list[Seg] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise FormatError(f'{path}[{index}] must be an object')
        easing = int(_number(event.get('easingType', 1), 'easingType'))
        bezier = int(event.get('bezier', 0) or 0)
        if bezier or easing > 1:
            if not allow_easing_loss:
                raise FormatError(
                    f'{path}[{index}] uses easingType={easing}, bezier={bezier}; '
                    f'only linear events can be interpreted by this reader')
            # Feature extraction only needs note times and line speeds; an eased
            # move event is approximated by its endpoints and reported.
            warnings.append(f'{path}[{index}]: easingType={easing} approximated as linear')
        left = _number(event.get('easingLeft', 0.0), 'easingLeft')
        right = _number(event.get('easingRight', 1.0), 'easingRight')
        if abs(left) > 1e-9 or abs(right - 1.0) > 1e-9:
            if not allow_easing_loss:
                raise FormatError(f'{path}[{index}] uses a partial easing window')
            warnings.append(f'{path}[{index}]: partial easing window '
                            f'[{left}, {right}] approximated as linear')
        t0 = time_of_beat(from_triple(event['startTime']))
        t1 = time_of_beat(from_triple(event['endTime']))
        if t1 < t0:
            warnings.append(f'{path}[{index}]: end before start, dropped')
            continue
        segs.append(Seg(t0, t1, float(event.get('start', 0.0)) * scale,
                        float(event.get('end', 0.0)) * scale))
    segs.sort(key=lambda s: (s.t0, s.t1))
    return Track(segs)


def _sum_tracks(tracks: list[Track], warnings: list[str], path: str) -> Track:
    """Additively merge event layers (RPE layers stack)."""
    live = [t for t in tracks if t.segments()]
    if not live:
        return Track.constant(0.0)
    if len(live) == 1:
        return live[0]
    warnings.append(f'{path}: {len(live)} event layers summed')
    points = sorted({s.t0 for t in live for s in t.segments()} |
                    {s.t1 for t in live for s in t.segments()})
    if len(points) < 2:
        return Track.constant(sum(t.at(points[0]) for t in live))
    segs = []
    for a, b in zip(points, points[1:]):
        if b - a < 1e-12:
            continue
        va = sum(t.at(a) for t in live)
        vb = sum(t.at(b) for t in live)
        segs.append(Seg(a, b, va, vb))
    return Track(segs)


def load_rpe(data: dict | str | bytes, *, source: str = '',
             allow_easing_loss: bool = False) -> Chart:
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
    if 'BPMList' not in payload or 'judgeLineList' not in payload:
        raise FormatError('not an RPE chart: missing BPMList or judgeLineList')

    warnings: list[str] = []
    axis = BpmAxis(payload.get('BPMList') or [])
    meta = payload.get('META') or {}
    offset = -float(meta.get('offset', 0.0)) / 1000.0

    raw_lines = payload['judgeLineList']
    if not isinstance(raw_lines, list):
        raise FormatError('judgeLineList must be a list')

    lines: list[Line] = []
    for index, raw in enumerate(raw_lines):
        if not isinstance(raw, dict):
            raise FormatError(f'judgeLineList[{index}] must be an object')
        path = f'judgeLineList[{index}]'
        factor = _number(raw.get('bpmfactor', 1.0), f'{path}.bpmfactor')
        if factor <= 0:
            raise FormatError(f'{path}.bpmfactor must be positive')
        if abs(factor - 1.0) > 1e-9:
            warnings.append(f'{path}: bpmfactor={factor:g} scales the global time axis')

        def line_time(beat: float) -> float:
            return axis.time(beat) * factor

        layers = raw.get('eventLayers') or []
        x_tracks, y_tracks, angle_tracks, alpha_tracks, speed_tracks = [], [], [], [], []
        for layer_index, layer in enumerate(layers):
            if layer is None:
                continue
            if not isinstance(layer, dict):
                raise FormatError(f'{path}.eventLayers[{layer_index}] must be an object')
            x_tracks.append(_linear_track(layer.get('moveXEvents') or [], line_time,
                                          1.0 / X_PX_PER_NORM, f'{path}.moveXEvents', warnings,
                                          allow_easing_loss))
            y_tracks.append(_linear_track(layer.get('moveYEvents') or [], line_time,
                                          1.0 / Y_PX_PER_NORM, f'{path}.moveYEvents', warnings,
                                          allow_easing_loss))
            angle_tracks.append(_linear_track(layer.get('rotateEvents') or [], line_time,
                                              1.0 / RPE_ROTATE_SIGN, f'{path}.rotateEvents', warnings,
                                              allow_easing_loss))
            alpha_tracks.append(_linear_track(layer.get('alphaEvents') or [], line_time,
                                              1.0 / RPE_ALPHA_MAX, f'{path}.alphaEvents', warnings,
                                              allow_easing_loss))
            speed_tracks.append(_linear_track(layer.get('speedEvents') or [], line_time,
                                              1.0 / OFFICIAL_SPEED_TO_RPE,
                                              f'{path}.speedEvents', warnings,
                                              allow_easing_loss))

        x_track = _sum_tracks(x_tracks, warnings, f'{path}.moveXEvents')
        y_track = _sum_tracks(y_tracks, warnings, f'{path}.moveYEvents')
        angle_track = _sum_tracks(angle_tracks, warnings, f'{path}.rotateEvents')
        alpha_track = _sum_tracks(alpha_tracks, warnings, f'{path}.alphaEvents')
        speed_track = _sum_tracks(speed_tracks, warnings, f'{path}.speedEvents')
        if not alpha_track.segments():
            warnings.append(f'{path}: no alpha events (line renders transparent)')
        if not speed_track.segments():
            warnings.append(f'{path}: no speed events, assuming 1 Y/s')
            speed_track = Track.constant(1.0)

        line = Line(bpm=axis.bpm_at_zero / factor, x=x_track, y=y_track, angle=angle_track,
                    alpha=alpha_track, speed=speed_track, notes=[], name=str(raw.get('Name', '')))

        for note_index, raw_note in enumerate(raw.get('notes') or []):
            where = f'{path}.notes[{note_index}]'
            if not isinstance(raw_note, dict):
                raise FormatError(f'{where} must be an object')
            raw_type = int(raw_note.get('type', 0))
            if raw_type not in RPE_TYPE_TO_OFFICIAL:
                raise FormatError(f'{where}.type {raw_type} is not 1..4')
            type_id = RPE_TYPE_TO_OFFICIAL[raw_type]
            start = line_time(from_triple(raw_note['startTime']))
            end = line_time(from_triple(raw_note.get('endTime', raw_note['startTime'])))
            hold = end - start if type_id == 3 else 0.0
            if type_id == 3 and hold <= 0:
                # same rule as the official reader: a zero-length hold is a tap
                warnings.append(f'{where}: hold without positive duration, treated as Tap')
                type_id, hold, end = 1, 0.0, start
            speed = _number(raw_note.get('speed', 1.0), f'{where}.speed')
            y_offset = _number(raw_note.get('yOffset', 0.0), f'{where}.yOffset')
            note = Note(
                type_id=type_id,
                t=start,
                x=_number(raw_note.get('positionX', 0.0), f'{where}.positionX') / X_PX_PER_NORM,
                speed=(line.speed_at(start) if type_id == 3 else speed),
                hold=hold,
                above=int(_number(raw_note.get('above', 1), f'{where}.above')) == 1,
                fake=int(_number(raw_note.get('isFake', 0), f'{where}.isFake')) != 0,
                floor_offset=y_offset / Y_PX_PER_OFFICIAL_UNIT,
            )
            if type_id == 3 and abs(speed - 1.0) > 1e-6:
                warnings.append(f'{where}: RPE hold speed {speed:g} cannot be represented '
                                f'in the official format (head speed is fixed)')
            line.notes.append(note)

        lines.append(line)

    chart = Chart(offset=offset, lines=lines, format_version=3, source=source,
                  meta={'rpe_version': meta.get('RPEVersion'),
                        'name': meta.get('name'), 'composer': meta.get('composer'),
                        'charter': meta.get('charter'), 'level': meta.get('level')},
                  warnings=warnings)
    chart.check()
    return chart

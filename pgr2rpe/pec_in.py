"""PEC (PE) text chart -> Re:PhiEdit (RPE) JSON.

PEC is Phira's text format, not a JSON dialect: the file extension tells you
nothing, only the content does.

Format reference (Phira Documents, ``chart-standard/chart-format/pe/``)
-----------------------------------------------------------------------
::

    <第一行>                                 offset，整数毫秒，计算时减去 175
    bp <beat> <bpm>

    cp <line> <beat> <x> <y>                 瞬时移动
    cm <line> <beat0> <beat1> <x> <y> <ease> 缓动移动（终点值；起点=当前值）
    cd <line> <beat> <deg>                   瞬时旋转
    cr <line> <beat0> <beat1> <deg> <ease>   缓动旋转
    ca <line> <beat> <alpha>                 瞬时透明度 0..255（-1.. 为扩展语义）
    cf <line> <beat0> <beat1> <alpha>        缓动透明度，只能线性
    cv <line> <beat> <speed>                 流速，缺省 10.0

    n1 <line> <beat> <x> <above> <fake>                 Tap（x 以 0 为中心）
    n2 <line> <beat0> <beat1> <x> <above> <fake>        Hold
    n3 <line> <beat> <x> <above> <fake>                 Flick
    n4 <line> <beat> <x> <above> <fake>                 Drag
    # <speed>   紧跟其后的续行（也可与音符同行的尾部）
    & <width>

Coordinates in the *file* are bottom-left based, 2048 x 1400, centre
``(1024, 700)``, +y up, and rotation is **clockwise positive** - the opposite of
both the official format and RPE. ``above == 1`` means "falls from above".

Conversion to RPE needs no time remapping at all: both formats count in beats
and both use a global BPM list, so ``bp`` becomes ``BPMList`` one to one. Only
five quantities change, and one of them is a trap:

* **event x/y** (``cp``/``cm``) use the 0..2048 / 0..1400 file canvas:
  ``x_rpe = (x - 1024) / 1024 * 675``, ``y_rpe = (y - 700) / 700 * 450``.
* **note x** (``n1..n4``) is *already centred on 0* (the docs' example is
  ``-40.000``), so ``x_rpe = x / 1024 * 675`` without subtracting the centre.
* **rotation** passes through unchanged: PEC files and RPE are both
  clockwise-positive (Phira negates both when reading).
* **speed** uses ``PEC_SPEED_TO_RPE`` (see below).
* **time** needs no conversion at all: both formats count in beats and both use
  a global BPM list, so ``bp`` becomes ``BPMList`` one to one.

The PEC speed unit is **not documented** (the docs only say "float, default
10.000").  Phira's PEC importer divides by 5.85 and its RPE importer multiplies
by ``10/45/HEIGHT_RATIO``; taking the documented RPE anchor (1 RPESpeed =
120 px/s on the 900-high canvas) as the common unit gives 0.641026.  A wrong
scale here would only change how far away notes are drawn, never when they
arrive, because arrival time is governed by the note's beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import FormatError, RPE_HEIGHT, RPE_WIDTH, to_triple

# --- documented PEC canvas -------------------------------------------------

PEC_CENTRE_X = 1024.0
PEC_CENTRE_Y = 700.0
PEC_HALF_WIDTH = 1024.0     # 2048 / 2
PEC_HALF_HEIGHT = 700.0     # 1400 / 2

PEC_X_TO_RPE = (RPE_WIDTH / 2.0) / PEC_HALF_WIDTH      # 675 / 1024
PEC_Y_TO_RPE = (RPE_HEIGHT / 2.0) / PEC_HALF_HEIGHT    # 450 / 700

# --- speed -----------------------------------------------------------------

PHIRA_PEC_SPEED_DIVISOR = 5.85
"""Phira's PEC importer: ``speed_events.push((time, value / 5.85))``."""

RPE_SPEED_IN_NORMALISED_PER_SECOND = 120.0 / (RPE_HEIGHT / 2.0)
"""1 RPESpeed = 120 px/s on a 900-high canvas = 0.26667 screen heights per second."""

PEC_SPEED_TO_RPE = (1.0 / PHIRA_PEC_SPEED_DIVISOR) / RPE_SPEED_IN_NORMALISED_PER_SECOND
"""0.641026: PEC speed -> RPESpeed, derived from the two documented anchors."""

OFFSET_BIAS_MS = 175
"""The docs: "实际计算时请减去 175".  Phira uses 150; ``offset_bias_ms`` overrides."""

DEFAULT_PEC_SPEED = 10.0
DEFAULT_RPE_VERSION = 170


class PecError(FormatError):
    """Raised when a PEC file cannot be interpreted."""


@dataclass
class PecEvent:
    t0: float
    t1: float
    value: float
    easing: int = 1

    @property
    def instant(self) -> bool:
        return abs(self.t1 - self.t0) < 1e-12


@dataclass
class PecNote:
    type_id: int            # PEC n1..n4 == 1 Tap / 2 Hold / 3 Flick / 4 Drag
    t: float
    x: float                # PEC px, centre based
    above: bool
    fake: bool
    speed: float = 1.0
    size: float = 1.0
    end: float = 0.0        # only for holds


@dataclass
class PecLine:
    x: list[PecEvent] = field(default_factory=list)
    y: list[PecEvent] = field(default_factory=list)
    angle: list[PecEvent] = field(default_factory=list)
    alpha: list[PecEvent] = field(default_factory=list)
    speed: list[tuple[float, float]] = field(default_factory=list)
    notes: list[PecNote] = field(default_factory=list)


@dataclass
class PecChart:
    offset_ms: float
    bpm: list[tuple[float, float]] = field(default_factory=list)
    lines: dict[int, PecLine] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: str = ''

    @property
    def last_beat(self) -> float:
        last = 0.0
        for line in self.lines.values():
            last = max(last, max((n.end or n.t) for n in line.notes) if line.notes else 0.0)
            for events in (line.x, line.y, line.angle, line.alpha):
                last = max(last, max((e.t1 for e in events), default=0.0))
            last = max(last, max((b for b, _ in line.speed), default=0.0))
        return last


def _float(token: str, where: str) -> float:
    try:
        return float(token)
    except (TypeError, ValueError) as exc:
        raise PecError(f'{where}: expected a number, got {token!r}') from exc


def _int(token: str, where: str) -> int:
    try:
        return int(float(token))
    except (TypeError, ValueError) as exc:
        raise PecError(f'{where}: expected an integer, got {token!r}') from exc


def _line_of(chart: PecChart, index: int) -> PecLine:
    if index < 0:
        raise PecError(f'negative judge line index {index}')
    if index >= 30 and index not in chart.lines:
        # The docs say PhiEditer stops at 30 lines, but real files do exceed it
        # (e.g. 10341.json uses index 30), so this is a warning, not an error.
        chart.warnings.append(f'judge line {index} exceeds the documented 30-line limit')
    return chart.lines.setdefault(index, PecLine())


def load_pec(text: str, *, source: str = '') -> PecChart:
    """Parse PEC text into raw PEC units (beats, PEC pixels, PEC degrees)."""
    if not isinstance(text, str):
        raise PecError('PEC input must be text')
    lines = text.splitlines()
    warnings: list[str] = []
    chart = PecChart(offset_ms=0.0, source=source)
    index = 0

    # Line 1 is the offset; tolerate a BOM and surrounding whitespace.
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        raise PecError('empty PEC file')
    first = lines[index].strip().lstrip('\ufeff')
    try:
        chart.offset_ms = float(first)
    except ValueError as exc:
        raise PecError(f'line 1 must be the integer offset, got {first!r}') from exc
    index += 1

    last_note: PecNote | None = None
    for lineno, raw in enumerate(lines[index:], start=index + 1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        command = parts[0]
        where = f'{source or "pec"}:{lineno}'

        if command == 'bp':
            if len(parts) < 3:
                raise PecError(f'{where}: bp needs <beat> <bpm>')
            beat = _float(parts[1], where)
            bpm = _float(parts[2], where)
            if bpm <= 0:
                raise PecError(f'{where}: bpm must be positive')
            chart.bpm.append((beat, bpm))
            continue

        if command in ('n1', 'n2', 'n3', 'n4'):
            if len(parts) < 6:
                raise PecError(f'{where}: {command} needs at least 6 fields')
            type_id = int(command[1])
            line_index = _int(parts[1], where)
            t0 = _float(parts[2], where)
            if type_id == 2:
                if len(parts) < 7:
                    raise PecError(f'{where}: n2 needs <end beat>')
                t1 = _float(parts[3], where)
                x_raw = _float(parts[4], where)
                above_raw = _int(parts[5], where)
                fake_raw = _int(parts[6], where)
                tail = parts[7:]
            else:
                if len(parts) < 6:
                    raise PecError(f'{where}: {command} needs <beat> <x> <above> <fake>')
                t1 = t0
                x_raw = _float(parts[3], where)
                above_raw = _int(parts[4], where)
                fake_raw = _int(parts[5], where)
                tail = parts[6:]
            if above_raw not in (1, 2):
                warnings.append(f'{where}: above={above_raw} is not 1/2, treated as below')
            # Note x is already centred on 0 (PEC note docs use -40.000), unlike
            # cp/cm which use the 0..2048 file canvas.
            note = PecNote(
                type_id=type_id, t=t0, x=x_raw,
                above=above_raw == 1, fake=fake_raw == 1, end=t1,
            )
            if type_id == 2 and t1 <= t0:
                warnings.append(f'{where}: hold ends before it starts, dropped')
                continue
            _line_of(chart, line_index).notes.append(note)
            last_note = note
            _apply_note_tail(note, tail, where, warnings)
            continue

        if command == '#':
            if last_note is None:
                warnings.append(f'{where}: "#" without a preceding note, ignored')
                continue
            if len(parts) < 2:
                warnings.append(f'{where}: empty "#" line, ignored')
                continue
            try:
                last_note.speed = float(parts[1])
            except ValueError:
                warnings.append(f'{where}: "#" payload {parts[1]!r} is not a number, '
                                f'treated as a comment')
            continue

        if command == '&':
            if last_note is None:
                warnings.append(f'{where}: "&" without a preceding note, ignored')
                continue
            if len(parts) < 2:
                warnings.append(f'{where}: empty "&" line, ignored')
                continue
            last_note.size = _float(parts[1], where)
            continue

        if command in ('cp', 'cm', 'cd', 'cr', 'ca', 'cf', 'cv'):
            if len(parts) < 4:
                raise PecError(f'{where}: {command} needs at least 4 fields')
            line_index = _int(parts[1], where)
            line = _line_of(chart, line_index)
            t0 = _float(parts[2], where)
            if command == 'cv':
                line.speed.append((t0, _float(parts[3], where)))
            elif command == 'cp':
                x = _float(parts[3], where)
                y = _float(parts[4], where) if len(parts) > 4 else PEC_CENTRE_Y
                line.x.append(PecEvent(t0, t0, x - PEC_CENTRE_X))
                line.y.append(PecEvent(t0, t0, y - PEC_CENTRE_Y))
            elif command == 'cm':
                if len(parts) < 7:
                    raise PecError(f'{where}: cm needs <beat0> <beat1> <x> <y> <ease>')
                t1 = _float(parts[3], where)
                line.x.append(PecEvent(t0, t1, _float(parts[4], where) - PEC_CENTRE_X,
                                       _int(parts[6], where)))
                line.y.append(PecEvent(t0, t1, _float(parts[5], where) - PEC_CENTRE_Y,
                                       _int(parts[6], where)))
            elif command == 'cd':
                line.angle.append(PecEvent(t0, t0, _float(parts[3], where)))
            elif command == 'cr':
                if len(parts) < 6:
                    raise PecError(f'{where}: cr needs <beat0> <beat1> <deg> <ease>')
                t1 = _float(parts[3], where)
                line.angle.append(PecEvent(t0, t1, _float(parts[4], where),
                                           _int(parts[5], where)))
            elif command == 'ca':
                line.alpha.append(PecEvent(t0, t0, _float(parts[3], where)))
            elif command == 'cf':
                if len(parts) < 5:
                    raise PecError(f'{where}: cf needs <beat0> <beat1> <alpha>')
                t1 = _float(parts[3], where)
                # cf has no easing parameter: the docs say it is always linear.
                line.alpha.append(PecEvent(t0, t1, _float(parts[4], where), 1))
            continue

        warnings.append(f'{where}: unknown command {command!r}, line skipped')

    if not chart.bpm:
        warnings.append('no bp line: assuming 120 BPM from beat 0')
        chart.bpm.append((0.0, 120.0))
    chart.bpm.sort()
    if chart.bpm[0][0] > 0:
        chart.bpm.insert(0, (0.0, chart.bpm[0][1]))
    chart.warnings.extend(warnings)
    return chart


def _apply_note_tail(note: PecNote, tail: list[str], where: str, warnings: list[str]) -> None:
    """Handle ``# speed`` / ``& width`` written on the same line as the note."""
    index = 0
    while index < len(tail):
        token = tail[index]
        if token in ('#', '&') and index + 1 < len(tail):
            value = _float(tail[index + 1], where)
            if token == '#':
                note.speed = value
            else:
                note.size = value
            index += 2
        else:
            warnings.append(f'{where}: trailing token {token!r} ignored')
            index += 1


def _resolve(events: list[PecEvent], initial: float, end_beat: float,
             warnings: list[str],
             where: str) -> list[tuple[float, float, float, float, int]]:
    """Turn PEC events into contiguous, explicit ``(t0, t1, v0, v1, easing)``.

    PEC's eased events only give the *end* value; the start value is whatever the
    line currently holds, and the docs also state that an eased event before any
    concrete value is invalid. Overlapping events are clipped the way Phira's
    ``sanitize_events`` does (sorted by end, then start).
    """
    prepared = sorted(events, key=lambda e: (e.t1, e.t0))
    segments: list[tuple[float, float, float, float, int]] = []
    cursor = 0.0
    value = initial
    for event in prepared:
        if event.t1 < event.t0 - 1e-9:
            warnings.append(f'{where}: event ends before it starts ({event.t0:.6f} -> '
                            f'{event.t1:.6f}), dropped')
            continue
        start = event.t0
        if start < cursor:
            warnings.append(f'{where}: overlapping event clipped to {cursor:.6f}')
            start = cursor
        if start > cursor:
            # Hold the previous value until the event starts, so that RPE does
            # not interpolate across the gap.
            segments.append((cursor, start, value, value, 1))
        if event.t1 > start:
            segments.append((start, event.t1, value, event.value, event.easing))
        cursor = max(cursor, event.t1)
        value = event.value
    if not segments:
        segments.append((0.0, end_beat, value, value, 1))
    elif cursor < end_beat:
        # Trailing constant segment: without it RPE would clamp to the value
        # before a final instant event and silently drop that jump.
        segments.append((cursor, end_beat, value, value, 1))
    return segments


def _initial_value(events: list[PecEvent], fallback: float) -> float:
    """PEC holds the first event's value backwards in time (Phira's Anim does)."""
    if not events:
        return fallback
    return min(events, key=lambda e: e.t0).value


def _event(t0: float, t1: float, v0: float, v1: float, easing: int) -> dict[str, Any]:
    return {
        'startTime': to_triple(t0),
        'endTime': to_triple(t1),
        'start': round(v0, 9),
        'end': round(v1, 9),
        'easingType': int(easing) if easing else 1,
        'easingLeft': 0.0,
        'easingRight': 1.0,
        'linkgroup': 0,
        'bezier': 0,
        'bezierPoints': [0.0, 0.0, 0.0, 0.0],
    }


def pec_to_rpe(chart: PecChart, *, meta: dict[str, Any] | None = None,
               rpe_version: int = DEFAULT_RPE_VERSION,
               offset_bias_ms: float = OFFSET_BIAS_MS) -> dict[str, Any]:
    """Serialise a parsed PEC chart as RPE JSON."""
    warnings = chart.warnings
    # One extra beat of padding so that a final instantaneous event can still be
    # expressed as a real segment instead of being clamped away.
    end_beat = max(chart.last_beat, 1.0) + 1.0
    offset_ms = offset_bias_ms - chart.offset_ms
    if abs(offset_ms - round(offset_ms)) > 1e-6:
        warnings.append(f'offset {offset_ms!r}ms quantised to an integer (RPE stores ms)')

    metadata = {
        'name': 'Converted chart', 'composer': '', 'charter': '', 'level': '',
        'song': '', 'background': 'background.png', 'illustration': 'illustration.png', 'id': 0,
    }
    for key, value in (meta or {}).items():
        metadata[key] = value
    metadata['RPEVersion'] = int(rpe_version)
    metadata['offset'] = int(round(offset_ms))

    line_count = max(chart.lines) + 1 if chart.lines else 0
    lines_out = []
    for index in range(line_count):
        line = chart.lines.get(index) or PecLine()

        x_segments = _resolve(line.x, _initial_value(line.x, 0.0), end_beat, warnings,
                              f'line {index}.x')
        y_segments = _resolve(line.y, _initial_value(line.y, 0.0), end_beat, warnings,
                              f'line {index}.y')
        angle_segments = _resolve(line.angle, _initial_value(line.angle, 0.0), end_beat,
                                  warnings, f'line {index}.angle')
        alpha_events = _resolve(line.alpha, _initial_value(line.alpha, 0.0), end_beat,
                                warnings, f'line {index}.alpha')

        move_x = [_event(t0, t1, v0 * PEC_X_TO_RPE, v1 * PEC_X_TO_RPE, e)
                  for t0, t1, v0, v1, e in x_segments]
        move_y = [_event(t0, t1, v0 * PEC_Y_TO_RPE, v1 * PEC_Y_TO_RPE, e)
                  for t0, t1, v0, v1, e in y_segments]
        # PEC file angles and RPE angles are both clockwise-positive, so the
        # value passes through unchanged (Phira negates both when reading).
        rotate = [_event(t0, t1, v0, v1, e) for t0, t1, v0, v1, e in angle_segments]
        alpha = [_event(t0, t1, v0, v1, e) for t0, t1, v0, v1, e in alpha_events]

        speed = _speed_events(line.speed, end_beat, warnings, index)

        notes_out = []
        for note in sorted(line.notes, key=lambda n: (n.t, not n.above, n.x)):
            start = to_triple(note.t)
            notes_out.append({
                'type': note.type_id,
                'above': 1 if note.above else 0,
                'startTime': start,
                'endTime': to_triple(note.end) if note.type_id == 2 else start,
                'positionX': round(note.x * PEC_X_TO_RPE, 9),
                'alpha': 255,
                'size': round(note.size, 9),
                'speed': round(note.speed, 9),
                'isFake': 1 if note.fake else 0,
                'visibleTime': 999999.0,
                'yOffset': 0.0,
            })

        lines_out.append({
            'Group': 0,
            'Name': f'line{index}',
            'Texture': 'line.png',
            'anchor': [0.5, 0.5],
            'posControl': [], 'sizeControl': [], 'skewControl': [],
            'yControl': [], 'alphaControl': [],
            'eventLayers': [{
                'alphaEvents': alpha,
                'moveXEvents': move_x,
                'moveYEvents': move_y,
                'rotateEvents': rotate,
                'speedEvents': speed,
            }],
            'extended': {
                'colorEvents': [], 'textEvents': [], 'scaleXEvents': [], 'scaleYEvents': [],
                'inclineEvents': [], 'paintEvents': [], 'gifEvents': [],
            },
            'father': -1,
            'rotateWithFather': False,
            'isCover': 0,
            'zOrder': 0,
            'bpmfactor': 1.0,
            'notes': notes_out,
            'numOfNotes': sum(1 for n in line.notes if n.type_id != 2),
        })

    return {
        'BPMList': [{'bpm': round(bpm, 9), 'startTime': to_triple(beat)}
                    for beat, bpm in chart.bpm],
        'META': metadata,
        'chartTime': round(_seconds_at(end_beat, chart.bpm), 3),
        'judgeLineGroup': ['A'],
        'judgeLineList': lines_out,
        'multiLineString': '',
        'multiScale': 1.0,
        'xybind': False,
    }


def _speed_events(speed: list[tuple[float, float]], end_beat: float,
                  warnings: list[str], line_index: int) -> list[dict]:
    """PEC ``cv`` sets the speed instantly; RPE wants constant segments.

    Phira inserts ``(0, 0)`` when the first ``cv`` is after beat 0, so a line is
    stationary until its first speed command - reproduced here.
    """
    # Two cv commands on the same beat: the later line wins, so keep only the
    # last value per beat (otherwise a zero-length segment would be emitted).
    collapsed: dict[float, float] = {}
    for beat, value in speed:
        collapsed[beat] = value
    events = sorted(collapsed.items())
    if not events:
        warnings.append(f'line {line_index}: no cv events, PEC/Phira leaves the line at '
                        f'speed 0')
        return [_event(0.0, end_beat, 0.0, 0.0, 1)]
    if events[0][0] > 0:
        events.insert(0, (0.0, 0.0))
    out = []
    for index, (beat, value) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else max(beat, end_beat)
        v = value * PEC_SPEED_TO_RPE
        out.append(_event(beat, end, v, v, 1))
    return out


def _seconds_at(beat: float, bpm_list: list[tuple[float, float]]) -> float:
    """Integrate the PEC BPM list from beat 0 to ``beat``."""
    total = 0.0
    for index, (start, bpm) in enumerate(bpm_list):
        end = bpm_list[index + 1][0] if index + 1 < len(bpm_list) else beat
        if beat <= start:
            break
        total += (min(beat, end) - start) * 60.0 / bpm
        if beat <= end:
            break
    return total


def load_pec_file(path: str) -> PecChart:
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        return load_pec(handle.read(), source=path)

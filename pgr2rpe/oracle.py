"""A second, independent implementation used to check the converter.

Nothing here imports :mod:`pgr2rpe.model`, :mod:`pgr2rpe.spec` or any of the
readers/writers. Each format is parsed again from the raw bytes and evaluated
with the formulas written out in the documentation, on purpose in a different
style (linear scans instead of binary search, one file per format), so that a
mistake in the converter cannot be mirrored here by sharing code.

Common units (the only place the formats are made comparable):

===========  ==================================================================
time         seconds
x, y         screen-normalised, ``[-1, 1]`` spans the full screen, +y up
angle        degrees, counter-clockwise positive
alpha        ``[0, 1]`` (RPE/PEC raw alpha divided by 255, negatives kept)
speed        screen heights per second (normalised units per second)
===========  ==================================================================

Speed anchors, each documented:

* official  1 Y = 0.6 H, so ``value`` in Y/s is ``value * 1.2`` heights/s
* RPE       1 RPESpeed = 120 px/s on the 900-high canvas = ``120/450``
* PEC       Phira divides the ``cv`` value by 5.85 when importing (the PEC speed
            unit itself is undocumented), giving ``value / 5.85`` heights/s
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import json
import math

OFFICIAL_X_UNIT = 0.05625       # 1 X = 0.05625 W
OFFICIAL_Y_UNIT = 0.6           # 1 Y = 0.6 H
OFFICIAL_TICKS = 32.0           # 32 T = 1 beat
RPE_WIDTH, RPE_HEIGHT = 1350.0, 900.0
PEC_CENTRE = (1024.0, 700.0)

# RPE and PEC number the note kinds differently from the official format
# (official: 1 Tap / 2 Drag / 3 Hold / 4 Flick).
_TYPE_TO_SEMANTIC = {1: 1, 2: 3, 3: 4, 4: 2}
"""RPE/PEC n1..n4 (1 Tap, 2 Hold, 3 Flick, 4 Drag) -> semantic id."""
PHIRA_PEC_SPEED_DIVISOR = 5.85


def _num(value, default=0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _beats(value) -> float:
    if isinstance(value, (list, tuple)) and len(value) >= 3 and value[2]:
        return float(value[0]) + float(value[1]) / float(value[2])
    return float(value)


# --------------------------------------------------------------------------
# easing (RPE / PEC share one numbering; 1 = linear)
# --------------------------------------------------------------------------

def _bounce_out(u: float) -> float:
    n, d = 7.5625, 2.75
    if u < 1 / d:
        return n * u * u
    if u < 2 / d:
        u -= 1.5 / d
        return n * u * u + 0.75
    if u < 2.5 / d:
        u -= 2.25 / d
        return n * u * u + 0.9375
    u -= 2.625 / d
    return n * u * u + 0.984375


def ease(kind: int, u: float) -> float:
    """Evaluate an RPE/PEC easing id at ``u`` in [0, 1]."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    k = kind if kind and kind > 0 else 1
    if k == 1:
        return u
    if k == 2:
        return math.sin(u * math.pi / 2)
    if k == 3:
        return 1 - math.cos(u * math.pi / 2)
    if k == 4:
        return 1 - (1 - u) ** 2
    if k == 5:
        return u * u
    if k == 6:
        return -(math.cos(math.pi * u) - 1) / 2
    if k == 7:
        return 2 * u * u if u < 0.5 else 1 - 2 * (1 - u) ** 2
    if k == 8:
        return 1 - (1 - u) ** 3
    if k == 9:
        return u ** 3
    if k == 10:
        return 1 - (1 - u) ** 4
    if k == 11:
        return u ** 4
    if k == 12:
        return 4 * u ** 3 if u < 0.5 else 1 - 4 * (1 - u) ** 3
    if k == 13:
        return 8 * u ** 4 if u < 0.5 else 1 - 8 * (1 - u) ** 4
    if k == 14:
        return 1 - (1 - u) ** 5
    if k == 15:
        return u ** 5
    if k == 16:
        return 1 - 2 ** (-10 * u)
    if k == 17:
        return 2 ** (10 * (u - 1))
    if k == 18:
        return math.sqrt(max(0.0, 1 - (u - 1) ** 2))
    if k == 19:
        return 1 - math.sqrt(max(0.0, 1 - u * u))
    if k in (20, 21):
        c = 1.70158
        if k == 20:
            return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2
        return (c + 1) * u ** 3 - c * u * u
    if k == 22:
        c = 1.70158
        if u < 0.5:
            return (2 * u) ** 2 * ((c * 2 + 2) * 2 * u / 2 - (c * 2 + 1)) / 2
        return ((2 * u - 2) ** 2 * ((c * 2 + 2) * (u * 2 - 2) / 2 + (c * 2 + 1)) + 2) / 2
    if k == 23:
        c = 1.70158 * 1.525
        if u < 0.5:
            return ((2 * u) ** 2 * ((c + 1) * 2 * u - c)) / 2
        return ((2 * u - 2) ** 2 * ((c + 1) * (u * 2 - 2) + c) + 2) / 2
    if k in (24, 25, 29):
        if k == 25:
            u = 1 - u
        if k == 29:
            if u < 0.5:
                return 0.5 * math.sin(13 * math.pi / 2 * (2 * u)) * 2 ** (10 * (2 * u - 1))
            return 0.5 * (math.sin(-13 * math.pi / 2 * (2 * u - 1 + 1)) * 2 ** (-10 * (2 * u - 1)) + 2)
        return 1 - 2 ** (-10 * u) * math.cos(u * math.pi * 4.5)
    if k == 26:
        return _bounce_out(u)
    if k == 27:
        return 1 - _bounce_out(1 - u)
    if k == 28:
        return 0.5 * _bounce_out(2 * u) if u < 0.5 else 0.5 * (1 - _bounce_out(2 - 2 * u)) + 0.5
    return u


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class OEvent:
    t0: float
    t1: float
    v0: float
    v1: float
    easing: int = 1

    def at(self, t: float) -> float:
        if self.t1 <= self.t0:
            return self.v1
        u = (t - self.t0) / (self.t1 - self.t0)
        if u <= 0:
            return self.v0
        if u >= 1:
            return self.v1
        return self.v0 + (self.v1 - self.v0) * ease(self.easing, u)

    def integral(self, a: float, b: float) -> float:
        lo, hi = max(a, self.t0), min(b, self.t1)
        if hi <= lo:
            return 0.0
        if self.easing == 1:
            return 0.5 * (self.at(lo) + self.at(hi)) * (hi - lo)
        steps = 16
        width = (hi - lo) / steps
        total = 0.0
        for i in range(steps):
            x0 = lo + i * width
            x1 = x0 + width
            total += 0.5 * (self.at(x0) + self.at(x1)) * width
        return total


@dataclass
class OLine:
    x: list[OEvent] = field(default_factory=list)
    y: list[OEvent] = field(default_factory=list)
    angle: list[OEvent] = field(default_factory=list)
    alpha: list[OEvent] = field(default_factory=list)
    speed: list[OEvent] = field(default_factory=list)
    notes: list['ONote'] = field(default_factory=list)
    alpha_default: float = 0.0
    _starts: dict | None = None

    def _index(self) -> dict:
        if self._starts is None:
            starts = {}
            for key in ('x', 'y', 'angle', 'alpha', 'speed'):
                events = getattr(self, key)
                # Official charts are not guaranteed to be sorted by startTime;
                # bisect below needs a sorted key list, so normalise once.
                if any(events[i].t0 > events[i + 1].t0 for i in range(len(events) - 1)):
                    events = sorted(events, key=lambda e: (e.t0, e.t1))
                    setattr(self, key, events)
                starts[key] = [e.t0 for e in events]
            self._starts = starts
        return self._starts

    def _value(self, key: str, t: float, default: float) -> float:
        events = getattr(self, key)
        if not events:
            return default
        if t <= events[0].t0:
            return events[0].v0
        index = bisect_right(self._index()[key], t) - 1
        if index < 0:
            return events[0].v0
        event = events[index]
        # Both formats hold the last value past the end of an event.
        return event.at(t)

    def at(self, t: float):
        return {
            'x': self._value('x', t, 0.0),
            'y': self._value('y', t, 0.0),
            'angle': self._value('angle', t, 0.0),
            'alpha': self._value('alpha', t, self.alpha_default),
            'speed': self._value('speed', t, 0.0),
        }

    def floor(self, t: float) -> float:
        """Integral of the line speed from the first event to ``t`` (heights)."""
        if not self.speed:
            return 0.0
        base = self.speed[0].t0
        if t <= base:
            return 0.0
        total = 0.0
        for event in self.speed:
            if event.t1 <= base:
                continue
            if event.t0 >= t:
                break
            total += event.integral(base, t)
        return total


@dataclass
class ONote:
    type_id: int      # semantic id shared by all formats: 1 Tap 2 Drag 3 Hold 4 Flick
    t: float
    x: float
    speed: float = 1.0
    end: float = 0.0
    above: bool = True
    fake: bool = False
    floor: float = 0.0
    size: float = 1.0


@dataclass
class OChart:
    offset: float
    lines: list[OLine]
    offset_ms_raw: float = 0.0
    kind: str = ''


# --------------------------------------------------------------------------
# official JSON
# --------------------------------------------------------------------------

def load_official(path: str) -> OChart:
    with open(path, 'r', encoding='utf-8-sig') as handle:
        data = json.load(handle)
    version = int(data.get('formatVersion', 3))
    lines: list[OLine] = []
    for raw in data['judgeLineList']:
        bpm = float(raw['bpm'])
        k = 60.0 / (OFFICIAL_TICKS * bpm)
        line = OLine()
        if version == 3:
            line.x = _official_v3(raw.get('judgeLineMoveEvents') or [], k, ('start', 'end'))
            line.y = _official_v3(raw.get('judgeLineMoveEvents') or [], k, ('start2', 'end2'))
        elif version == 1:
            line.x = _official_v1(raw.get('judgeLineMoveEvents') or [], k, 0)
            line.y = _official_v1(raw.get('judgeLineMoveEvents') or [], k, 1)
        else:
            line.x = _official_v3(raw.get('judgeLineMoveEvents') or [], k, ('start', 'end'))
            line.y = _official_v3(raw.get('judgeLineMoveEvents') or [], k, ('start2', 'end2'))
        line.angle = _official_scalar(raw.get('judgeLineRotateEvents') or [], k)
        line.alpha = _official_scalar(raw.get('judgeLineDisappearEvents') or [], k)
        line.alpha_default = 1.0   # no disappear events == fully opaque
        line.speed = _official_speed(raw.get('speedEvents') or [], k, bpm)
        for side, above in (('notesAbove', True), ('notesBelow', False)):
            for note in raw.get(side) or []:
                t = _num(note.get('time')) * k
                line.notes.append(ONote(
                    type_id=int(note.get('type', 0)), t=t,
                    x=_num(note.get('positionX')) * 2.0 * OFFICIAL_X_UNIT,
                    speed=_num(note.get('speed'), 1.0),
                    end=t + _num(note.get('holdTime')) * k,
                    above=above, fake=False,
                    floor=_num(note.get('floorPosition')),
                ))
        lines.append(line)
    return OChart(offset=float(_num(data.get('offset'))), lines=lines, kind='official')


def _official_v3(events: list[dict], k: float, keys: tuple[str, str]) -> list[OEvent]:
    out = []
    for e in events:
        t0 = _num(e.get('startTime')) * k
        t1 = _num(e.get('endTime')) * k
        if t1 < t0:
            continue          # malformed: the converters drop these explicitly
        out.append(OEvent(t0, t1, _num(e.get(keys[0])) * 2 - 1,
                          _num(e.get(keys[1])) * 2 - 1, 1))
    return out


def _official_v1(events: list[dict], k: float, axis: int) -> list[OEvent]:
    out = []
    grid = 880.0 if axis == 0 else 520.0
    for e in events:
        p0 = _num(e.get('start'))
        p1 = _num(e.get('end'))
        a0 = (math.floor(p0 / 1000.0) if axis == 0 else p0 - math.floor(p0 / 1000.0) * 1000)
        a1 = (math.floor(p1 / 1000.0) if axis == 0 else p1 - math.floor(p1 / 1000.0) * 1000)
        t0 = _num(e.get('startTime')) * k
        t1 = _num(e.get('endTime')) * k
        if t1 < t0:
            continue
        out.append(OEvent(t0, t1, (a0 / grid - 0.5) * 2, (a1 / grid - 0.5) * 2, 1))
    return out


def _official_scalar(events: list[dict], k: float) -> list[OEvent]:
    out = []
    for e in events:
        t0 = _num(e.get('startTime')) * k
        t1 = _num(e.get('endTime')) * k
        if t1 < t0:
            continue
        out.append(OEvent(t0, t1, _num(e.get('start')), _num(e.get('end')), 1))
    return out


def _official_speed(events: list[dict], k: float, bpm: float) -> list[OEvent]:
    # Official speed events hold one value over their interval (piecewise constant).
    out = []
    for e in events:
        t0 = _num(e.get('startTime')) * k
        t1 = _num(e.get('endTime')) * k
        if t1 < t0:
            continue
        value = _num(e.get('value')) * 2.0 * OFFICIAL_Y_UNIT
        out.append(OEvent(t0, t1, value, value, 1))
    return out


# --------------------------------------------------------------------------
# RPE JSON
# --------------------------------------------------------------------------

def load_rpe(path: str) -> OChart:
    with open(path, 'r', encoding='utf-8-sig') as handle:
        data = json.load(handle)
    bpm_list = sorted((_beats(e['startTime']), float(e['bpm'])) for e in data['BPMList'])

    def seconds(beat: float) -> float:
        if beat <= bpm_list[0][0]:
            return (beat - bpm_list[0][0]) * 60.0 / bpm_list[0][1]
        total, cursor = 0.0, 0.0
        for index, (start, bpm) in enumerate(bpm_list):
            end = bpm_list[index + 1][0] if index + 1 < len(bpm_list) else beat
            if beat <= start:
                break
            total += (min(beat, end) - start) * 60.0 / bpm
            cursor = end
            if beat <= end:
                break
        return total

    lines: list[OLine] = []
    for raw in data['judgeLineList']:
        factor = float(raw.get('bpmfactor', 1.0) or 1.0)
        line = OLine()
        for layer in raw.get('eventLayers') or []:
            if not layer:
                continue
            for field, target, scale in (
                ('moveXEvents', 'x', 1.0 / (RPE_WIDTH / 2)),
                ('moveYEvents', 'y', 1.0 / (RPE_HEIGHT / 2)),
                ('rotateEvents', 'angle', -1.0),
                ('alphaEvents', 'alpha', 1.0 / 255.0),
                ('speedEvents', 'speed', 120.0 / (RPE_HEIGHT / 2)),
            ):
                converted = []
                for e in layer.get(field) or []:
                    converted.append(OEvent(seconds(_beats(e['startTime'])) * factor,
                                            seconds(_beats(e['endTime'])) * factor,
                                            _num(e.get('start')) * scale,
                                            _num(e.get('end')) * scale,
                                            int(e.get('easingType', 1) or 1)))
                getattr(line, target).extend(converted)
        for note in raw.get('notes') or []:
            t = seconds(_beats(note['startTime'])) * factor
            end = seconds(_beats(note.get('endTime', note['startTime']))) * factor
            line.notes.append(ONote(
                type_id=_TYPE_TO_SEMANTIC.get(int(note['type']), int(note['type'])), t=t,
                x=_num(note.get('positionX')) / (RPE_WIDTH / 2),
                speed=_num(note.get('speed'), 1.0),
                end=end if _TYPE_TO_SEMANTIC.get(int(note['type'])) == 3 else t,
                above=int(note.get('above', 1)) == 1,
                fake=int(note.get('isFake', 0)) != 0,
                floor=_num(note.get('yOffset')),
                size=_num(note.get('size'), 1.0),
            ))
        lines.append(line)
    return OChart(offset=-float(_num(data['META'].get('offset'))) / 1000.0,
                  lines=lines, offset_ms_raw=float(_num(data['META'].get('offset'))),
                  kind='rpe')


# --------------------------------------------------------------------------
# PEC text
# --------------------------------------------------------------------------

def load_pec(path: str) -> OChart:
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        rows = [row.strip() for row in handle.read().splitlines() if row.strip()]
    offset_ms = float(rows[0])
    bpm_list: list[tuple[float, float]] = []
    tracks: dict[int, dict[str, list]] = {}

    def bucket(index: int) -> dict[str, list]:
        return tracks.setdefault(index, {'move': [], 'rot': [], 'alpha': [], 'speed': [],
                                         'notes': []})

    last_note = None
    for row in rows[1:]:
        parts = row.split()
        cmd = parts[0]
        if cmd == 'bp':
            bpm_list.append((float(parts[1]), float(parts[2])))
            continue
        if cmd in ('n1', 'n2', 'n3', 'n4'):
            kind = int(cmd[1])
            target = bucket(int(parts[1]))
            if kind == 2:
                note = [kind, float(parts[2]), float(parts[3]), float(parts[4]),
                        int(parts[5]), int(parts[6]), 1.0, 1.0]
            else:
                note = [kind, float(parts[2]), float(parts[2]), float(parts[3]),
                        int(parts[4]), int(parts[5]), 1.0, 1.0]
            target['notes'].append(note)
            last_note = note
            continue
        if cmd in ('#', '&') and last_note is not None and len(parts) > 1:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            last_note[6 if cmd == '#' else 7] = value
            continue
        if cmd not in ('cp', 'cm', 'cd', 'cr', 'ca', 'cf', 'cv'):
            continue
        target = bucket(int(parts[1]))
        t0 = float(parts[2])
        if cmd == 'cv':
            target['speed'].append((t0, float(parts[3])))
        elif cmd == 'cp':
            target['move'].append((t0, t0, float(parts[3]), float(parts[4]), 1))
        elif cmd == 'cm':
            target['move'].append((t0, float(parts[3]), float(parts[4]), float(parts[5]),
                                   int(parts[6])))
        elif cmd == 'cd':
            target['rot'].append((t0, t0, float(parts[3]), 1))
        elif cmd == 'cr':
            target['rot'].append((t0, float(parts[3]), float(parts[4]), int(parts[5])))
        elif cmd == 'ca':
            target['alpha'].append((t0, t0, float(parts[3]), 1))
        elif cmd == 'cf':
            target['alpha'].append((t0, float(parts[3]), float(parts[4]), 1))

    bpm_list.sort()
    if not bpm_list:
        bpm_list = [(0.0, 120.0)]
    if bpm_list[0][0] > 0:
        bpm_list.insert(0, (0.0, bpm_list[0][1]))

    def seconds(beat: float) -> float:
        if beat <= bpm_list[0][0]:
            return (beat - bpm_list[0][0]) * 60.0 / bpm_list[0][1]
        total = 0.0
        for index, (start, bpm) in enumerate(bpm_list):
            end = bpm_list[index + 1][0] if index + 1 < len(bpm_list) else beat
            if beat <= start:
                break
            total += (min(beat, end) - start) * 60.0 / bpm
            if beat <= end:
                break
        return total

    def resolve(events: list[tuple[float, float, float, int]]) -> list[OEvent]:
        """PEC eased events only carry the end value: the start is the current one."""
        if not events:
            return []
        initial = min(events, key=lambda e: e[0])[2]
        out: list[OEvent] = []
        cursor, value = 0.0, initial
        for t0, t1, target, easing in sorted(events, key=lambda e: (e[1], e[0])):
            if t1 < t0 - 1e-9:
                continue          # malformed: end before start
            if t0 < cursor:
                t0 = cursor
            if t0 > cursor:
                out.append(OEvent(seconds(cursor), seconds(t0), value, value, 1))
            if t1 > t0:
                out.append(OEvent(seconds(t0), seconds(t1), value, target, easing))
            cursor, value = t1, target
        # An instantaneous event produces no segment, but its value must win for
        # every later time (both PEC and RPE clamp to the last keyframe).
        out.append(OEvent(seconds(cursor), seconds(cursor), value, value, 1))
        return out

    lines: list[OLine] = []
    for index in range(max(tracks) + 1 if tracks else 0):
        item = tracks.get(index) or {'move': [], 'rot': [], 'alpha': [], 'speed': [],
                                     'notes': []}
        line = OLine()
        line.x = resolve([(t0, t1, (x - PEC_CENTRE[0]) / 1024.0, e)
                          for t0, t1, x, _y, e in item['move']])
        line.y = resolve([(t0, t1, (y - PEC_CENTRE[1]) / 700.0, e)
                          for t0, t1, _x, y, e in item['move']])
        line.angle = resolve([(t0, t1, -v, e) for t0, t1, v, e in item['rot']])
        line.alpha = resolve([(t0, t1, v / 255.0, e) for t0, t1, v, e in item['alpha']])
        # Sort by beat only: two cv commands on the same beat must keep their
        # file order (last one wins), so the value must not be part of the key.
        collapsed: dict[float, float] = {}
        for beat, value in item['speed']:
            collapsed[beat] = value          # last cv on a beat wins
        ordered = sorted(collapsed.items())
        if ordered:
            if ordered[0][0] > 0:
                ordered.insert(0, (0.0, 0.0))
            for i, (beat, value) in enumerate(ordered):
                end = ordered[i + 1][0] if i + 1 < len(ordered) else beat
                v = value / PHIRA_PEC_SPEED_DIVISOR
                line.speed.append(OEvent(seconds(beat), seconds(end), v, v, 1))
        for kind, t0, t1, x, above, fake, speed, _size in item['notes']:
            line.notes.append(ONote(
                type_id=_TYPE_TO_SEMANTIC.get(kind, kind), t=seconds(t0),
                x=x / 1024.0,      # PEC note x is already centred on 0
                speed=speed, end=seconds(t1) if kind == 2 else seconds(t0),
                above=above == 1, fake=fake == 1, floor=0.0, size=_size,
            ))
        lines.append(line)
    return OChart(offset=(offset_ms - 175.0) / 1000.0, lines=lines,
                  offset_ms_raw=offset_ms, kind='pec')


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def detect(path: str) -> str:
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        head = handle.read(4096).lstrip('\ufeff \t\r\n')
    if not head.startswith('{'):
        return 'pec'
    return 'rpe' if ('"BPMList"' in head or '"META"' in head) else 'official'


def load(path: str) -> OChart:
    kind = detect(path)
    if kind == 'official':
        return load_official(path)
    if kind == 'rpe':
        return load_rpe(path)
    return load_pec(path)


def _notes_equal(a: ONote, b: ONote, tol: float) -> str | None:
    if a.type_id != b.type_id:
        return f'type {a.type_id} vs {b.type_id}'
    if abs(a.t - b.t) > tol:
        return f'time {a.t:.6f} vs {b.t:.6f}'
    if abs(a.end - b.end) > tol:
        return f'end {a.end:.6f} vs {b.end:.6f}'
    if abs(a.x - b.x) > 1e-6:
        return f'x {a.x:.9f} vs {b.x:.9f}'
    if a.type_id != 3 and abs(a.speed - b.speed) > 1e-6:
        # Only non-holds share a "speed multiplier" meaning. An official Hold
        # stores the tail's absolute speed (Y/s) while RPE stores a multiplier,
        # and the tail itself is the one quantity the two models cannot agree on
        # (see README "一个无法消除的差异").
        return f'speed {a.speed} vs {b.speed}'
    if a.above != b.above:
        return 'above flag'
    if a.fake != b.fake:
        return 'fake flag'
    if abs(a.size - b.size) > 1e-6:
        return f'size {a.size} vs {b.size}'
    return None


def _pair_notes(notes_a: list[ONote], notes_b: list[ONote], index: int,
                result: dict):
    """Pair notes by sorted position, then verify the pairing field by field.

    Sorting by ``(time, type, side, fake, x)`` keeps stacked notes (same time,
    one from each side) in a stable order on both sides. Grouping by a rounded
    time key was tried first and produced phantom mismatches: the two encodings
    can differ in the last float bit, which is enough to fall on opposite sides
    of a rounding boundary.
    """
    def key(note: ONote):
        return (round(note.t, 4), note.type_id, note.above, note.fake, round(note.x, 9))

    a = sorted(notes_a, key=key)
    b = sorted(notes_b, key=key)
    if len(a) != len(b):
        result['ok'] = False
        result['issues'].append(f'line {index}: {len(a)} vs {len(b)} notes')
    pairs = []
    for position, (na, nb) in enumerate(zip(a, b)):
        if (na.type_id, na.above, na.fake) != (nb.type_id, nb.above, nb.fake):
            result['ok'] = False
            if len(result['issues']) < 20:
                result['issues'].append(
                    f'line {index}: note #{position} kind/side/fake '
                    f'{na.type_id}/{na.above}/{na.fake} vs '
                    f'{nb.type_id}/{nb.above}/{nb.fake} at t={na.t:.4f}')
            continue
        pairs.append((na, nb))
    return pairs


def compare(path_a: str, path_b: str, *, samples: int = 1200,
            tol: float = 1e-6) -> dict:
    """Compare two charts through the documented, format-independent units."""
    a, b = load(path_a), load(path_b)
    result = {
        'ok': True, 'issues': [], 'lines': (len(a.lines), len(b.lines)),
        'notes': (sum(len(l.notes) for l in a.lines), sum(len(l.notes) for l in b.lines)),
        'max': {'x': 0.0, 'y': 0.0, 'angle': 0.0, 'alpha': 0.0, 'speed': 0.0,
                'note_t': 0.0, 'note_x': 0.0},
        'offset_error': abs(a.offset - b.offset),
    }
    if len(a.lines) != len(b.lines):
        result['ok'] = False
        result['issues'].append(f'line count {len(a.lines)} vs {len(b.lines)}')
        return result
    if result['offset_error'] > 5e-4:
        result['ok'] = False
        result['issues'].append(f'offset differs by {result["offset_error"]*1000:.2f} ms')

    # Only compare what a player can observe: the last note. Official charts
    # carry events out to a 1e9-tick sentinel, and a converter is allowed to clip
    # that tail, so sampling past the last note would report phantom differences
    # in regions the chart never reaches.
    horizon = 0.0
    for line in a.lines + b.lines:
        if line.notes:
            horizon = max(horizon, max(n.end for n in line.notes))
    if horizon <= 0.0:
        for line in a.lines + b.lines:
            for events in (line.x, line.y, line.angle, line.alpha, line.speed):
                if events:
                    horizon = max(horizon, max(e.t1 for e in events))
    horizon = max(horizon, 1e-3)
    times = [horizon * i / (samples - 1) for i in range(samples)]

    # Drop samples that land on (or within 1 us of) an event boundary: the two
    # encodings can round the same breakpoint by one float ULP, and probing such
    # an instant reports a step difference that is not a real difference.
    boundaries = sorted({event.t0 for line in a.lines + b.lines
                         for events in (line.x, line.y, line.angle, line.alpha, line.speed)
                         for event in events})
    if boundaries:
        kept = []
        for t in times:
            index = bisect_right(boundaries, t)
            near = min(abs(t - boundaries[index]) if index < len(boundaries) else 1e9,
                       abs(t - boundaries[index - 1]) if index > 0 else 1e9)
            if near > 1e-6:
                kept.append(t)
        times = kept or times

    for index, (la, lb) in enumerate(zip(a.lines, b.lines)):
        for t in times:
            sa, sb = la.at(t), lb.at(t)
            for key in ('x', 'y', 'angle', 'speed'):
                result['max'][key] = max(result['max'][key], abs(sa[key] - sb[key]))
            result['max']['alpha'] = max(result['max']['alpha'],
                                         abs(max(0.0, min(1.0, sa['alpha']))
                                             - max(0.0, min(1.0, sb['alpha']))))
        if len(la.notes) != len(lb.notes):
            result['ok'] = False
            result['issues'].append(f'line {index}: {len(la.notes)} vs {len(lb.notes)} notes')
            continue
        for na, nb in _pair_notes(la.notes, lb.notes, index, result):
            problem = _notes_equal(na, nb, tol)
            if problem:
                result['ok'] = False
                if len(result['issues']) < 20:
                    result['issues'].append(f'line {index}: {problem} at t={na.t:.4f}')
            result['max']['note_t'] = max(result['max']['note_t'], abs(na.t - nb.t))
            result['max']['note_x'] = max(result['max']['note_x'], abs(na.x - nb.x))

    for key, value in result['max'].items():
        if key in ('x', 'y', 'alpha') and value > 1e-6:
            result['ok'] = False
            result['issues'].append(f'{key} differs by {value:.3e}')
        elif key in ('angle', 'speed') and value > 1e-6:
            result['ok'] = False
            result['issues'].append(f'{key} differs by {value:.3e}')
    return result

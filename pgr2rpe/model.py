"""Canonical chart model.

Units (all physical, no encoding leakage):

* ``time``        seconds, chart-local playhead time (tick 0 == 0 s)
* ``x``, ``y``    screen-normalised, ``[-1, 1]`` maps to the full screen,
                  +x right, +y up, (0, 0) is the screen centre
* ``angle``       degrees, counter-clockwise positive
* ``alpha``       0..1
* ``speed``       official Y per second
* note ``x``      same normalised space as line ``x``
* note ``speed``  dimensionless multiplier (see :mod:`pgr2rpe.spec`)

Keeping the model in one unit system is what makes the round trip checkable:
the official reader and writer are almost the identity, and the RPE reader and
writer apply the exact factors from :mod:`pgr2rpe.spec` in opposite directions.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

from .spec import EPS, FormatError


@dataclass(frozen=True)
class Seg:
    """One linear segment ``v0 -> v1`` over ``[t0, t1]`` (seconds).

    A constant track (speed events) uses ``v0 == v1``. A zero-length segment
    (``t0 == t1``) is a step: the value jumps to ``v1`` at ``t0``.
    """

    t0: float
    t1: float
    v0: float
    v1: float

    def at(self, t: float) -> float:
        if self.t1 <= self.t0:
            return self.v1
        u = (t - self.t0) / (self.t1 - self.t0)
        u = min(1.0, max(0.0, u))
        return self.v0 + (self.v1 - self.v0) * u

    def integral(self, a: float, b: float) -> float:
        """Integral of this segment over ``[a, b]`` (clipped to the segment)."""
        lo = max(a, self.t0)
        hi = min(b, self.t1)
        if hi <= lo:
            return 0.0
        if self.t1 <= self.t0:
            return 0.0
        v_lo = self.at(lo)
        v_hi = self.at(hi)
        return 0.5 * (v_lo + v_hi) * (hi - lo)


class Track:
    """A piecewise-linear function of time with clamping outside its domain.

    Both formats clamp: before the first event the first event's start value is
    used, after the last event the last event's end value is used. Official
    charts encode this by making the first event start at a huge negative tick
    and the last end at 1e9; RPE readers clamp by construction.
    """

    __slots__ = ('segs', 'default', '_starts', '_cum')

    def __init__(self, segs: Iterable[Seg] = (), default: float = 0.0):
        raw = sorted(segs, key=lambda s: (s.t0, s.t1))
        resolved: list[Seg] = []
        for seg in raw:
            if not (math.isfinite(seg.t0) and math.isfinite(seg.t1)):
                raise FormatError('event time must be finite')
            if seg.t1 < seg.t0:
                raise FormatError(f'event ends before it starts: {seg}')
            while resolved and seg.t0 < resolved[-1].t1:
                last = resolved[-1]
                if seg.t0 <= last.t0:
                    resolved.pop()
                    continue
                resolved[-1] = Seg(last.t0, seg.t0, last.v0, last.at(seg.t0))
            resolved.append(seg)
        self.segs: tuple[Seg, ...] = tuple(resolved)
        self.default = float(default)
        self._starts: list[float] | None = None
        self._cum: list[float] | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def constant(cls, value: float) -> 'Track':
        return cls((), default=float(value))

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float]]) -> 'Track':
        """Build a track from ``[(t, value), ...]`` measured at segment edges."""
        pts = list(points)
        return cls(Seg(pts[i][0], pts[i + 1][0], pts[i][1], pts[i + 1][1])
                   for i in range(len(pts) - 1))

    @classmethod
    def from_steps(cls, steps: Sequence[tuple[float, float, float]]) -> 'Track':
        """Build a track from ``[(t0, t1, value), ...]`` constant segments."""
        return cls(Seg(t0, t1, v, v) for t0, t1, v in steps)

    # -- queries -----------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.segs)

    @property
    def t_start(self) -> float:
        return self.segs[0].t0 if self.segs else 0.0

    @property
    def t_end(self) -> float:
        return self.segs[-1].t1 if self.segs else 0.0

    def at(self, t: float) -> float:
        if not self.segs:
            return self.default
        if t <= self.segs[0].t0:
            return self.segs[0].v0
        if t >= self.segs[-1].t1:
            return self.segs[-1].v1
        lo, hi = 0, len(self.segs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.segs[mid].t0 <= t:
                lo = mid
            else:
                hi = mid - 1
        return self.segs[lo].at(t)

    def integral(self, t: float, origin: float | None = None) -> float:
        """Integral from ``origin`` (default: start of the first segment) to ``t``.

        Uses a lazily built prefix sum so a note-distance sweep over a chart with
        thousands of keyframes stays O(n log n) instead of O(n^2).
        """
        if not self.segs:
            return 0.0
        base = self.segs[0].t0 if origin is None else origin
        if t <= base:
            return 0.0
        if self._starts is None:
            starts = [seg.t0 for seg in self.segs]
            cumulative = [0.0]
            for seg in self.segs:
                cumulative.append(cumulative[-1] + seg.integral(seg.t0, seg.t1))
            self._starts, self._cum = starts, cumulative
        starts, cumulative = self._starts, self._cum
        index = bisect_right(starts, t) - 1
        if index < 0:
            return 0.0
        seg = self.segs[index]
        total = cumulative[index] + seg.integral(seg.t0, min(t, seg.t1))
        if base > seg.t0:
            total -= self.integral_partial(base)
        return total

    def integral_partial(self, t: float) -> float:
        """Integral from the first segment start to ``t``."""
        return self.integral(t, origin=self.segs[0].t0) if self.segs else 0.0

    def is_constant(self, tol: float = EPS) -> bool:
        if not self.segs:
            return True
        values = [self.at(self.segs[0].t0), self.at(self.segs[-1].t1)]
        for seg in self.segs:
            values.append(seg.v0)
            values.append(seg.v1)
        return max(values) - min(values) <= tol

    # -- transformation ----------------------------------------------------

    def clipped(self, w0: float, w1: float) -> 'Track':
        """Restrict the domain to ``[w0, w1]`` without changing values inside.

        Segments crossing a boundary keep their slope: the boundary value is
        evaluated and used as the new end point, so the function is identical on
        ``[w0, w1]`` and clamped outside exactly as before.
        """
        if not self.segs:
            return self
        if self.t_end <= w0 or self.t_start >= w1:
            return Track((), default=self.at(min(max(w0, self.t_start), self.t_end)))
        out: list[Seg] = []
        for seg in self.segs:
            t0, t1 = seg.t0, seg.t1
            if t1 < w0 or t0 > w1:
                continue
            v0 = seg.at(t0)
            v1 = seg.at(t1)
            if t0 < w0:
                t0 = w0
                v0 = seg.at(w0)
            if t1 > w1:
                t1 = w1
                v1 = seg.at(w1)
            out.append(Seg(t0, t1, v0, v1))
        if not out:
            return Track((), default=self.at(w0))
        return Track(out)

    def segments(self) -> tuple[Seg, ...]:
        return self.segs

    def compressed(self, tol: float = 1e-12) -> 'Track':
        """Merge contiguous collinear segments (identical function, fewer events)."""
        out: list[Seg] = []
        for seg in self.segs:
            if out:
                last = out[-1]
                contiguous = abs(seg.t0 - last.t1) <= tol
                joined = abs(last.v1 - seg.v0) <= tol
                both_real = last.t1 > last.t0 and seg.t1 > seg.t0
                if contiguous and joined and both_real:
                    slope_a = (last.v1 - last.v0) / (last.t1 - last.t0)
                    slope_b = (seg.v1 - seg.v0) / (seg.t1 - seg.t0)
                    if abs(slope_a - slope_b) <= tol * max(1.0, abs(slope_a)):
                        out[-1] = Seg(last.t0, seg.t1, last.v0, seg.v1)
                        continue
            out.append(seg)
        return Track(out)

    def clamp_values(self, lo: float, hi: float) -> 'Track':
        """Clamp the function to ``[lo, hi]``, splitting segments at the crossings.

        Clamping is not linear: clamping only the two end points of a segment
        would turn an over-range ramp into a slower ramp. The official renderer
        clamps the alpha it samples, so the track has to be split where it
        crosses a bound.
        """
        out: list[Seg] = []
        for seg in self.segs:
            if seg.t1 <= seg.t0:
                out.append(Seg(seg.t0, seg.t1,
                               min(hi, max(lo, seg.v0)), min(hi, max(lo, seg.v1))))
                continue
            span = seg.v1 - seg.v0
            cuts: list[float] = []
            if span != 0.0:
                for bound in (lo, hi):
                    u = (bound - seg.v0) / span
                    if 0.0 < u < 1.0:
                        cuts.append(u)
            cuts = sorted(set(cuts))
            previous_u = 0.0
            previous_v = min(hi, max(lo, seg.v0))
            for u in cuts + [1.0]:
                value = min(hi, max(lo, seg.v0 + span * u))
                out.append(Seg(seg.t0 + (seg.t1 - seg.t0) * previous_u,
                               seg.t0 + (seg.t1 - seg.t0) * u,
                               previous_v, value))
                previous_u, previous_v = u, value
        return Track(out)


@dataclass
class Note:
    """One note, times in seconds, position normalised."""

    type_id: int
    t: float
    x: float
    speed: float = 1.0
    hold: float = 0.0
    above: bool = True
    fake: bool = False
    floor_offset: float = 0.0
    """``floorPosition - P_J(time)`` in Y units.

    Stored as a residual, not as an absolute position, because the two formats
    anchor their vertical integrals at different origins: an absolute value
    would not survive a round trip, a residual does.
    """

    @property
    def t_end(self) -> float:
        return self.t + self.hold

    @property
    def is_hold(self) -> bool:
        # The converter's canonical model uses official IDs, where 3 is Hold.
        # Analysis code restores native RPE IDs after reading and must use
        # ``type_id == 2`` there instead of this converter property.
        return self.type_id == 3


@dataclass
class Line:
    bpm: float
    x: Track = field(default_factory=lambda: Track.constant(0.0))
    y: Track = field(default_factory=lambda: Track.constant(0.0))
    angle: Track = field(default_factory=lambda: Track.constant(0.0))
    alpha: Track = field(default_factory=lambda: Track.constant(1.0))
    speed: Track = field(default_factory=lambda: Track.constant(1.0))
    notes: list[Note] = field(default_factory=list)
    name: str = ''

    def floor_at(self, t: float) -> float:
        """Official ``P_J(t)``: integral of the line speed, in Y units."""
        return self.speed.integral(t)

    def speed_at(self, t: float) -> float:
        return self.speed.at(t)


@dataclass
class Chart:
    offset: float
    """Seconds. Official: positive delays the chart. RPE equivalent is -ms."""

    lines: list[Line] = field(default_factory=list)
    format_version: int = 3
    source: str = ''
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- bounds ------------------------------------------------------------

    def playable_notes(self) -> list[Note]:
        return [n for line in self.lines for n in line.notes if not n.fake]

    @property
    def first_note_time(self) -> float:
        notes = self.playable_notes()
        return min((n.t for n in notes), default=0.0)

    @property
    def last_note_time(self) -> float:
        notes = self.playable_notes()
        return max((n.t_end for n in notes), default=0.0)

    @property
    def max_bpm(self) -> float:
        return max((line.bpm for line in self.lines), default=120.0)

    def dominant_bpm(self) -> float:
        """BPM of the line holding the most playable notes (ties: first line)."""
        best_bpm, best_count = 120.0, -1
        for line in self.lines:
            count = sum(1 for n in line.notes if not n.fake)
            if count > best_count:
                best_bpm, best_count = line.bpm, count
        return best_bpm

    def time_window(self, margin: float = 10.0) -> tuple[float, float]:
        """Domain over which the two encodings must agree.

        Everything before 0 s is unobservable (the chart starts at time 0 with
        the offset applied). The window starts at most at 0 and ends after the
        last playable note so that trailing events are preserved.
        """
        w0 = min(0.0, self.first_note_time) - margin
        w1 = max(self.last_note_time, 0.0) + margin
        return w0, w1

    def note_count(self) -> int:
        return sum(1 for line in self.lines for n in line.notes)

    def check(self) -> None:
        for index, line in enumerate(self.lines):
            if not math.isfinite(line.bpm) or line.bpm <= 0:
                raise FormatError(f'judgeLineList[{index}] has invalid bpm {line.bpm!r}')
            for note in line.notes:
                if not math.isfinite(note.t):
                    raise FormatError(f'judgeLineList[{index}] note has non-finite time')
                if note.is_hold and note.hold <= 0:
                    raise FormatError(f'judgeLineList[{index}] hold note without positive length')
                if not note.is_hold and note.hold != 0:
                    raise FormatError(f'judgeLineList[{index}] non-hold note carries hold time')
                if note.type_id not in (1, 2, 3, 4):
                    raise FormatError(f'judgeLineList[{index}] unknown note type {note.type_id}')

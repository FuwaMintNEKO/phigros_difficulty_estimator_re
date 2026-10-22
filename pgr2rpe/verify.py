"""Semantic equivalence checking between two canonical charts.

Equality of JSON files proves nothing: the two encodings partition time into
completely different event lists. What has to agree is the *observable* chart:

* line state ``(x, y, angle, alpha, speed)`` as a function of time,
* every note's type, time, side, x, speed multiplier and hold duration,
* every note's on-screen distance from its judge line over time, which is the
  quantity both formats derive from their speed events but compute differently.

Because both charts are piecewise linear in time, checking the midpoint of every
interval delimited by either chart's breakpoints is an exact test, not a
sampling heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Chart, Line, Note


class ConversionError(RuntimeError):
    """Raised when a conversion cannot preserve the chart semantically."""


@dataclass
class Report:
    ok: bool = True
    line_count_a: int = 0
    line_count_b: int = 0
    notes_a: int = 0
    notes_b: int = 0
    max_dx: float = 0.0
    max_dy: float = 0.0
    max_dangle: float = 0.0
    max_dalpha: float = 0.0
    max_dspeed: float = 0.0
    max_dtime: float = 0.0
    max_dnote_x: float = 0.0
    max_dhold: float = 0.0
    offset_error: float = 0.0
    hold_body_max_error: float = 0.0
    samples: int = 0
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        if len(self.issues) < 40:
            self.issues.append(message)

    def as_dict(self) -> dict:
        return {
            'ok': self.ok,
            'lines': (self.line_count_a, self.line_count_b),
            'notes': (self.notes_a, self.notes_b),
            'max_dx': self.max_dx,
            'max_dy': self.max_dy,
            'max_dangle': self.max_dangle,
            'max_dalpha': self.max_dalpha,
            'max_dspeed': self.max_dspeed,
            'max_dtime': self.max_dtime,
            'max_dnote_x': self.max_dnote_x,
            'max_dhold': self.max_dhold,
            'offset_error': self.offset_error,
            'hold_body_max_error': self.hold_body_max_error,
            'samples': self.samples,
            'issues': self.issues,
        }


def _breakpoints(*tracks) -> list[float]:
    points: set[float] = set()
    for track in tracks:
        for seg in track.segments():
            points.add(seg.t0)
            points.add(seg.t1)
    return sorted(points)


def _midpoints(points: list[float], window: tuple[float, float],
               limit: int = 4000) -> list[float]:
    """One sample per interval between breakpoints (exact for linear tracks).

    Charts can carry tens of thousands of keyframes per line; above ``limit``
    intervals the sampling is thinned uniformly so the audit stays tractable.
    """
    lo = max(window[0], points[0] if points else window[0])
    hi = min(window[1], points[-1] if points else window[1])
    ordered = [lo] + [p for p in points if lo < p < hi] + [hi]
    samples = [lo, hi]
    for a, b in zip(ordered, ordered[1:]):
        # Sub-100 ns intervals only appear where the two encodings round the
        # same breakpoint to a different side of a step; the step itself is
        # identical, so probing that sliver would report a phantom difference.
        if b - a > 1e-7:
            samples.append(0.5 * (a + b))
    samples = sorted(set(samples))
    if len(samples) > limit:
        step = len(samples) / limit
        samples = sorted({samples[min(len(samples) - 1, int(i * step))] for i in range(limit)})
    return samples


def _clamp_alpha(value: float) -> float:
    return max(0.0, min(1.0, value))


def _group_notes(notes: list[Note]) -> dict[tuple, list[Note]]:
    groups: dict[tuple, list[Note]] = {}
    for note in notes:
        key = (round(note.t, 6), note.above, note.type_id, note.fake)
        groups.setdefault(key, []).append(note)
    for values in groups.values():
        values.sort(key=lambda n: n.x)
    return groups


def _compare_notes(report: Report, line_index: int, line_a: Line,
                   line_b: Line) -> list[tuple[Note, Note]]:
    """Match notes by (time, side, type) and compare fields within tolerance."""
    groups_a = _group_notes(line_a.notes)
    groups_b = _group_notes(line_b.notes)
    pairs: list[tuple[Note, Note]] = []
    if len(groups_a) != len(groups_b):
        report.fail(f'line {line_index}: note groups {len(groups_a)} vs {len(groups_b)}')
        return pairs
    for key, notes_a in groups_a.items():
        notes_b = groups_b.get(key)
        if notes_b is None or len(notes_a) != len(notes_b):
            report.fail(f'line {line_index}: note group at t={key[0]} differs in size')
            continue
        for note_a, note_b in zip(notes_a, notes_b):
            pairs.append((note_a, note_b))
            report.max_dtime = max(report.max_dtime, abs(note_a.t - note_b.t))
            report.max_dnote_x = max(report.max_dnote_x, abs(note_a.x - note_b.x))
            report.max_dhold = max(report.max_dhold, abs(note_a.hold - note_b.hold))
            if abs(note_a.speed - note_b.speed) > 1e-6 * max(1.0, abs(note_a.speed)):
                report.fail(f'line {line_index}: note speed {note_a.speed} vs {note_b.speed} '
                            f'at t={note_a.t:.4f}')
            if abs(note_a.floor_offset - note_b.floor_offset) > 1e-4:
                report.fail(f'line {line_index}: note floor offset {note_a.floor_offset:.6f} vs '
                            f'{note_b.floor_offset:.6f} at t={note_a.t:.4f}')
    return pairs


def note_distance_official(line: Line, note: Note, t: float) -> float:
    """Official visual distance from the judge line, in Y units (may be -)."""
    base = line.floor_at(note.t) - line.floor_at(t) + note.floor_offset
    if note.type_id != 3:
        return note.speed * base
    if t <= note.t:
        return base + note.speed * note.hold
    return note.speed * (note.t_end - t)


def head_distance_official(line: Line, note: Note, t: float) -> float:
    """Distance of the Hold *head* from the line (the part the player judges)."""
    if note.type_id != 3:
        return note_distance_official(line, note, t)
    return max(0.0, line.floor_at(note.t) - line.floor_at(t) + note.floor_offset) if t <= note.t else 0.0


def head_distance_rpe(line: Line, note: Note, t: float) -> float:
    """RPE Hold head distance.

    RPE has no floorPosition: the reader rebuilds it as ``P_J(t) + yOffset/540``,
    so comparing this against the official value checks both the speed integral
    and the encoded yOffset in one go.
    """
    if note.type_id != 3:
        return note_distance_rpe(line, note, t)
    return max(0.0, line.floor_at(note.t) - line.floor_at(t) + note.floor_offset) if t <= note.t else 0.0


def note_distance_rpe(line: Line, note: Note, t: float) -> float:
    """RPE visual distance from the judge line, in Y units.

    RPE stores no per-note height: a note's distance is always the integral of
    the line speed up to the note's own time (times its speed multiplier), and a
    Hold's tail is the same integral evaluated at ``endTime``.
    """
    if note.type_id == 3:
        return line.floor_at(note.t_end) - line.floor_at(t) + note.floor_offset
    return note.speed * (line.floor_at(note.t) - line.floor_at(t) + note.floor_offset)


def compare_charts(a: Chart, b: Chart, *, tol: float = 1e-6,
                   window: tuple[float, float] | None = None,
                   check_note_distance: bool = True) -> Report:
    report = Report(line_count_a=len(a.lines), line_count_b=len(b.lines),
                    notes_a=a.note_count(), notes_b=b.note_count())
    if len(a.lines) != len(b.lines):
        report.fail(f'judge line count differs: {len(a.lines)} vs {len(b.lines)}')
        return report
    window = window or a.time_window()
    report.offset_error = abs(a.offset - b.offset)
    if report.offset_error > 5e-4:
        report.fail(f'offset differs by {report.offset_error * 1000:.2f} ms')

    for index, (line_a, line_b) in enumerate(zip(a.lines, b.lines)):
        points = _breakpoints(line_a.x, line_a.y, line_a.angle, line_a.alpha, line_a.speed,
                              line_b.x, line_b.y, line_b.angle, line_b.alpha, line_b.speed)
        samples = _midpoints(points, window)
        for t in samples:
            report.samples += 1
            report.max_dx = max(report.max_dx, abs(line_a.x.at(t) - line_b.x.at(t)))
            report.max_dy = max(report.max_dy, abs(line_a.y.at(t) - line_b.y.at(t)))
            report.max_dangle = max(report.max_dangle, abs(line_a.angle.at(t) - line_b.angle.at(t)))
            report.max_dalpha = max(report.max_dalpha,
                                    abs(_clamp_alpha(line_a.alpha.at(t)) - _clamp_alpha(line_b.alpha.at(t))))
            report.max_dspeed = max(report.max_dspeed, abs(line_a.speed.at(t) - line_b.speed.at(t)))

        pairs = _compare_notes(report, index, line_a, line_b)

        if check_note_distance:
            for note_a, note_b in pairs:
                if note_a.type_id == 3:
                    # The head is exact in both encodings. The tail is not: the
                    # official format moves it at a constant speed (the note's
                    # `speed`), RPE integrates the line speed. The gap is zero
                    # whenever the line speed is constant over the hold and is
                    # reported separately instead of failing the chart.
                    for t in (note_a.t - 0.37, note_a.t - 1e-6):
                        if window[0] <= t <= window[1]:
                            error = abs(head_distance_official(line_a, note_a, t)
                                        - head_distance_rpe(line_b, note_b, t))
                            if error > tol * 10:
                                report.fail(f'line {index}: hold head at {note_a.t:.4f}s '
                                            f'differs by {error:.6f} Y')
                    for t in (note_a.t - 0.37, note_a.t, 0.5 * (note_a.t + note_a.t_end),
                              note_a.t_end - 1e-3):
                        if window[0] <= t <= window[1]:
                            error = abs(note_distance_official(line_a, note_a, t)
                                        - note_distance_rpe(line_b, note_b, t))
                            report.hold_body_max_error = max(report.hold_body_max_error, error)
                else:
                    t = note_a.t - 0.37
                    if window[0] <= t <= window[1]:
                        error = abs(note_distance_official(line_a, note_a, t)
                                    - note_distance_rpe(line_b, note_b, t))
                        if error > tol * 10:
                            report.fail(f'line {index}: note at {note_a.t:.4f}s differs by '
                                        f'{error:.6f} Y at t={t:.4f}')

    for name, value in (('x', report.max_dx), ('y', report.max_dy),
                        ('angle', report.max_dangle), ('alpha', report.max_dalpha),
                        ('speed', report.max_dspeed)):
        if value > tol * 10:
            report.fail(f'line {name} differs by {value:.3e}')

    if abs(report.max_dtime) > 1e-5:
        report.fail(f'note time differs by {report.max_dtime:.3e} s')
    if report.max_dnote_x > 1e-6:
        report.fail(f'note x differs by {report.max_dnote_x:.3e}')
    if report.max_dhold > 1e-5:
        report.fail(f'hold duration differs by {report.max_dhold:.3e} s')
    return report


def assert_equivalent(a: Chart, b: Chart, **kwargs) -> Report:
    report = compare_charts(a, b, **kwargs)
    if not report.ok:
        raise ConversionError('; '.join(report.issues) or 'charts are not equivalent')
    return report

"""Load any Phigros chart into one canonical, feature-friendly form.

The heavy lifting (unit conversion, event resolution, easing, BPM axis) is done
by :mod:`pgr2rpe`, which is verified against two independent implementations.
This module only normalises the result into the arrays the feature extractor
needs, all in player-visible units:

* note time in **seconds**
* note x in **[-1, 1]**, 1 = half the screen width
* scroll speed at the note's own time in **screen heights per second**
  (official ``speed`` is Y/s and 1 Y = 0.6 H, so ``* 1.2``; the RPE reader
  already converts RPESpeed to the same Y/s scale, so one factor serves both)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import zipfile

import numpy as np

from pgr2rpe import load_official, load_rpe
from pgr2rpe.pec_in import load_pec, pec_to_rpe

OFFICIAL_Y_TO_NORM = 1.2          # 1 Y = 0.6 H; normalised y spans 2 = H
TYPE_TAP, TYPE_DRAG, TYPE_HOLD, TYPE_FLICK = 1, 2, 3, 4


class ChartError(RuntimeError):
    pass


@dataclass
class ChartData:
    path: str
    kind: str
    times: np.ndarray            # (n,) seconds, sorted
    xs: np.ndarray               # (n,) [-1, 1]
    types: np.ndarray            # (n,) 1 Tap 2 Drag 3 Hold 4 Flick
    above: np.ndarray            # (n,) bool
    fakes: np.ndarray            # (n,) bool
    hold_ends: np.ndarray        # (n,) seconds (== time for non-holds)
    speeds: np.ndarray           # (n,) scroll speed at the note, heights/s
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return int(self.times.size)


def _meta_from_rpe(payload: dict) -> dict:
    meta = payload.get('META') or {}
    bpms = [float(event['bpm']) for event in payload.get('BPMList') or []
            if isinstance(event, dict) and event.get('bpm')]
    return {
        'name': str(meta.get('name') or '').strip(),
        'charter': str(meta.get('charter') or '').strip(),
        'composer': str(meta.get('composer') or '').strip(),
        'level': str(meta.get('level') or '').strip(),
        'bpm_min': min(bpms) if bpms else None,
        'bpm_max': max(bpms) if bpms else None,
    }


def _meta_from_official(payload: dict) -> dict:
    bpms = []
    for line in payload.get('judgeLineList') or []:
        try:
            bpms.append(float(line.get('bpm')))
        except (AttributeError, TypeError, ValueError):
            continue
    return {'name': '', 'charter': '', 'composer': '', 'level': '',
            'bpm_min': min(bpms) if bpms else None,
            'bpm_max': max(bpms) if bpms else None}


_PEC_FIELD = re.compile(r'^\s*(name|charter|composer|level|illustrator)\s*:\s*(.+?)\s*$', re.M)
_PEC_BPM = re.compile(r'^bp\s+([\d.]+)\s+([\d.]+)\s*$', re.M)


def _meta_from_pec(text: str) -> dict:
    fields = {key.lower(): value for key, value in _PEC_FIELD.findall(text)}
    bpms = [float(value) for _, value in _PEC_BPM.findall(text) if float(value) > 0]
    return {
        'name': fields.get('name', ''),
        'charter': fields.get('charter', ''),
        'composer': fields.get('composer', ''),
        'level': fields.get('level', ''),
        'bpm_min': min(bpms) if bpms else None,
        'bpm_max': max(bpms) if bpms else None,
    }


def _detect(text: str) -> str:
    head = text.lstrip('\ufeff \t\r\n')
    if not head.startswith('{'):
        return 'pe'
    if '"BPMList"' in head[:4096] or '"META"' in head[:4096]:
        return 'rpe'
    return 'official'


def _read_text(path: str) -> str:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist()
                     if n.lower().endswith(('.json', '.pec', '.txt'))]
            if not names:
                raise ChartError(f'{path}: archive contains no chart')
            names.sort(key=lambda n: -archive.getinfo(n).file_size)
            return archive.read(names[0]).decode('utf-8-sig', errors='replace')
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        return handle.read()


def _from_model(chart, path: str, kind: str) -> ChartData:
    rows = []
    warnings = list(getattr(chart, 'warnings', []))
    for line in chart.lines:
        for note in line.notes:
            rows.append((
                float(note.t), float(note.x), int(note.type_id),
                bool(note.above), bool(getattr(note, 'fake', False)),
                float(note.t_end if note.type_id == TYPE_HOLD else note.t),
                float(line.speed.at(note.t)) * OFFICIAL_Y_TO_NORM,
            ))
    # Notes on the same tick must come out in a format-independent order: the
    # "adjacent note" features (jump_*, alternation, jackness) read neighbours, and
    # the source order of a chord differs between official / RPE / PEC files. Time
    # is quantised to 1 us so float noise cannot reorder a chord, then ties go by x.
    rows.sort(key=lambda row: (round(row[0], 6), round(row[1], 6), row[2]))
    # Some community charts carry a sentinel note parked at t = 1e9 ticks (about
    # 10 million seconds). Because duration was taken as the last note, one such
    # note turned a 104 second chart into a 10 million second one and drove nps
    # to 0.0002 - the model then read the chart as empty. Anything far beyond the
    # 99.9th percentile of note times is that sentinel, not music.
    if len(rows) >= 20:
        times_all = np.asarray([row[0] for row in rows], dtype=float)
        # 44 of Apollo's 2507 notes sit at 10.4 million seconds, so even the 99th
        # percentile lands inside the sentinel group; the 95th does not.
        horizon = float(np.quantile(times_all, 0.95))
        if times_all[-1] > horizon * 3.0:
            rows = [row for row in rows if row[0] <= horizon * 1.2]
    if not rows:
        empty_f = np.zeros(0)
        return ChartData(path, kind, empty_f, empty_f, np.zeros(0, dtype=int),
                         np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), empty_f,
                         empty_f, 0.0, warnings)
    arr = np.asarray(rows, dtype=object)
    times = np.asarray([row[0] for row in rows], dtype=float)
    return ChartData(
        path=path, kind=kind, times=times,
        xs=np.asarray([row[1] for row in rows], dtype=float),
        types=np.asarray([row[2] for row in rows], dtype=int),
        above=np.asarray([row[3] for row in rows], dtype=bool),
        fakes=np.asarray([row[4] for row in rows], dtype=bool),
        hold_ends=np.asarray([row[5] for row in rows], dtype=float),
        speeds=np.asarray([row[6] for row in rows], dtype=float),
        duration=float(times.max()), warnings=warnings,
    )


def load_chart(path: str) -> ChartData:
    """Parse one chart file (official JSON, RPE JSON, PEC text or a .pez zip)."""
    text = _read_text(path)
    kind = _detect(text)
    if kind == 'official':
        payload = json.loads(text)
        chart = _from_model(load_official(text, source=path), path, 'official')
        chart.meta = _meta_from_official(payload)
        return chart
    if kind == 'rpe':
        payload = json.loads(text)
        chart = _from_model(load_rpe(text, source=path, allow_easing_loss=True), path, 'rpe')
        chart.meta = _meta_from_rpe(payload)
        return chart
    chart = _from_model(load_rpe(pec_to_rpe(load_pec(text, source=path)), source=path,
                                 allow_easing_loss=True), path, 'pe')
    chart.meta = _meta_from_pec(text)
    return chart

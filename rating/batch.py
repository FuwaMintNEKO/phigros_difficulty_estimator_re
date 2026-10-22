"""Batch prediction over a folder of charts.

    python -m rating.batch <dir> [<dir> ...] --out predictions.csv --speed 1.0

Accepts the same things the web UI does: loose ``.json`` / ``.pec`` / ``.pez`` /
``.zip`` files, ``*.Charter.0`` packages (``chart.json`` + ``info.json``) and
folders nested one level deep. Every row carries the five category shares, so the
output can be sorted by "what kind of hard" and not only by the number.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile

from .compose import CATEGORIES
from .predictor import Predictor

CHART_SUFFIXES = ('.json', '.pec', '.pez', '.zip')
SKIP_NAMES = {'info.json', 'meta.json', 'music.json', 'chartmeta.json'}


def _zip_info(path: str) -> dict:
    """Read ``info.json`` from a chart package without unpacking it."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = next((n for n in archive.namelist() if n.endswith('info.json')), None)
            if name is None:
                return {}
            return json.loads(archive.read(name).decode('utf-8-sig'))
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError):
        return {}


def _info(folder: str) -> dict:
    path = os.path.join(folder, 'info.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8-sig') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def discover(root: str) -> list[tuple[str, dict]]:
    """(chart path, metadata) for every chart under ``root``."""
    found: list[tuple[str, dict]] = []

    def take(path: str, meta: dict) -> None:
        name = os.path.basename(path).lower()
        if name in SKIP_NAMES:
            return
        found.append((path, meta))

    if os.path.isfile(root):
        take(root, {})
        return found

    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            chart = os.path.join(full, 'chart.json')
            meta = _info(full)
            if os.path.exists(chart):
                take(chart, meta)
                continue
            for name in sorted(os.listdir(full)):
                if name.lower().endswith(CHART_SUFFIXES) and name.lower() not in SKIP_NAMES:
                    take(os.path.join(full, name), meta)
        elif entry.lower().endswith(CHART_SUFFIXES) and entry.lower() not in SKIP_NAMES:
            meta = _zip_info(full) if entry.lower().endswith(('.zip', '.pez')) else {}
            take(full, meta)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='rating.batch')
    parser.add_argument('roots', nargs='+', help='谱面文件或目录（可多个）')
    parser.add_argument('--out', required=True)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--model', default=None)
    parser.add_argument('--labels', default='',
                        help='可选：CSV（id,diff）按文件名 join，用于算误差')
    args = parser.parse_args(argv)

    predictor = Predictor(args.model) if args.model else Predictor()
    known: dict[str, float] = {}
    known_names: dict[str, str] = {}
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, 'r', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                value = (row.get('diff') or '').strip()
                if row.get('id'):
                    if value:
                        known[row['id']] = float(value)
                    if row.get('name'):
                        known_names[row['id']] = row['name']

    items: list[tuple[str, dict]] = []
    for root in args.roots:
        if not os.path.exists(root):
            print(f'! 路径不存在: {root}')
            continue
        items.extend(discover(root))

    seen, unique = set(), []
    for path, meta in items:
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, meta))
    items = unique
    print(f'发现 {len(items)} 张谱面，倍速 {args.speed:g}x', flush=True)

    rows, failed = [], []
    started = time.perf_counter()
    for index, (path, meta) in enumerate(items, 1):
        try:
            result = predictor.explain(path, args.speed)
        except Exception as exc:  # noqa: BLE001
            failed.append((path, f'{type(exc).__name__}: {exc}'))
            continue
        percent = result['percent']
        stem = os.path.splitext(os.path.basename(path))[0]
        label = meta.get('difficulty', '') or (f'{known[stem]:.1f}' if stem in known else '')
        # a folder called just "json" tells the reader nothing, so fall back to
        # the file stem (which for phira files is the chart id)
        parent = os.path.basename(os.path.dirname(path))
        display = (meta.get('name') or known_names.get(stem)
                   or (stem if parent.lower() in ('json', 'charts', '') else result['name']))
        if display.endswith('.Charter.0'):
            display = display[: -len('.Charter.0')]
        rows.append({
            'name': display,
            'official': f'{result["official"]:.2f}',
            'community': f'{result["community"]:.2f}',
            'notes': result['stats']['notes'],
            'duration': result['stats']['duration'],
            'format': result['kind'],
            **{f'{key}%': f'{percent.get(key, 0.0):.0f}' for key in CATEGORIES},
            'label': label,
            'err': (f'{float(result["community"]) - float(label):+.2f}' if label else ''),
            'level': meta.get('level', ''),
            'file': os.path.basename(path),
            'path': os.path.abspath(path),
        })
        if index % 50 == 0:
            print(f'  {index}/{len(items)}  {time.perf_counter()-started:.0f}s', flush=True)

    rows.sort(key=lambda row: -float(row['official']))
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scored = [row for row in rows if row.get('err')]
    if scored:
        errors = [float(row['err']) for row in scored]
        mae = sum(abs(e) for e in errors) / len(errors)
        print(f'{len(scored)} 张有参考定数：MAE {mae:.3f}，bias {sum(errors)/len(errors):+.3f}')
    print(f'\n{len(rows)} 张已预测 -> {args.out}')
    if failed:
        print(f'{len(failed)} 张失败：')
        for path, why in failed[:10]:
            print(f'  ! {os.path.basename(path)}: {why}')
    if args.speed != 1.0:
        print(f'（全部按 {args.speed:g}x 变速预测）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Batch conversion + verification over a chart corpus."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from .cli import detect_format
from .official_in import load_official
from .rpe_in import load_rpe
from .rpe_out import FLOOR_POLICY_AUTO, write_rpe
from .spec import FormatError
from .verify import compare_charts


def _iter_json(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('node_modules', '.git', '__pycache__')]
        for name in sorted(filenames):
            if name.lower().endswith('.json'):
                yield os.path.join(dirpath, name)


def convert_and_check(path: str, *, floor_policy: str = FLOOR_POLICY_AUTO,
                      bpm: float | None = None, output_root: str | None = None,
                      input_root: str | None = None,
                      verify: bool = True,
                      independent: bool = False) -> dict[str, Any]:
    """Convert one file in memory and return a result record.

    ``.json`` files may hold any of the three formats, so the format is decided
    by content; PEC files are verified with :mod:`pgr2rpe.oracle` because the
    model is linear-only, while official charts use the stricter in-model
    comparison.
    """
    record: dict[str, Any] = {'path': path}
    started = time.perf_counter()
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
            text = handle.read()
        try:
            kind = detect_format(text)
        except FormatError as exc:
            return {'path': path, 'skipped': f'not a chart ({exc})'}

        if kind == 'pe':
            return _convert_pec(path, text, record, started, output_root, input_root, verify)
        if kind != 'official':
            return {'path': path, 'skipped': kind}

        source = load_official(text, source=path)
        payload = write_rpe(source, bpm=bpm, floor_policy=floor_policy,
                            meta={'name': os.path.basename(path)})
        if output_root:
            relative = os.path.relpath(path, input_root or os.path.dirname(path))
            target = os.path.join(output_root, os.path.splitext(relative)[0] + '.rpe.json')
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
            record['output'] = target
        # Serialise and re-read so the check covers JSON round-trip effects
        # (triple quantisation, float formatting), not just the in-memory dict.
        produced_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        if verify and independent:
            import tempfile
            from . import oracle
            handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.json',
                                                 delete=False)
            handle.write(produced_text)
            handle.close()
            try:
                result = oracle.compare(path, handle.name)
            finally:
                os.unlink(handle.name)
            report = None
            record['ok'] = result['ok']
            record['kind'] = 'official'
            record.update({
                'lines': len(source.lines), 'notes': source.note_count(),
                'duration': round(source.last_note_time, 3), 'offset': source.offset,
                'bpm': round(source.dominant_bpm(), 6),
                'multi_bpm': len({round(line.bpm, 9) for line in source.lines}) > 1,
                'dmax': {key: result['max'][key] for key in
                         ('x', 'y', 'angle', 'alpha', 'speed', 'note_t', 'note_x')} |
                        {'hold': 0.0, 'offset': result['offset_error'], 'hold_body': 0.0},
                'warnings': source.warnings[:12], 'warning_count': len(source.warnings),
                'ignored': {re.sub(r'\[\d+\]$', '', k): v
                            for k, v in (source.meta.get('ignored_fields') or {}).items()},
                'issues': result['issues'][:6], 'rpe_bytes': len(produced_text),
                'ms': round((time.perf_counter() - started) * 1000, 1),
            })
            return record
        if verify:
            back = load_rpe(json.loads(produced_text), source=path + '::rpe')
            report = compare_charts(source, back)
        else:
            report = None
        record.update({
            'ok': report.ok if report else None,
            'kind': 'official',
            'lines': len(source.lines),
            'notes': source.note_count(),
            'duration': round(source.last_note_time, 3),
            'offset': source.offset,
            'bpm': round(source.dominant_bpm(), 6),
            'multi_bpm': len({round(line.bpm, 9) for line in source.lines}) > 1,
            'dmax': {
                'x': report.max_dx, 'y': report.max_dy, 'angle': report.max_dangle,
                'alpha': report.max_dalpha, 'speed': report.max_dspeed,
                'note_t': report.max_dtime, 'note_x': report.max_dnote_x,
                'hold': report.max_dhold, 'offset': report.offset_error,
                'hold_body': report.hold_body_max_error,
            } if report else {},
            'warnings': source.warnings[:12],
            'warning_count': len(source.warnings),
            'ignored': {re.sub(r'\[\d+\]$', '', k): v
                        for k, v in (source.meta.get('ignored_fields') or {}).items()},
            'issues': report.issues[:6] if report else [],
            'rpe_bytes': len(produced_text),
            'ms': round((time.perf_counter() - started) * 1000, 1),
        })
    except (FormatError, ValueError, OSError) as exc:
        record.update({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    return record


def _convert_pec(path: str, text: str, record: dict[str, Any], started: float,
                 output_root: str | None, input_root: str | None,
                 verify: bool) -> dict[str, Any]:
    from . import oracle
    from .pec_in import load_pec, pec_to_rpe

    source = load_pec(text, source=path)
    payload = pec_to_rpe(source, meta={'name': os.path.basename(path)})
    produced_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

    target = None
    if output_root:
        relative = os.path.relpath(path, input_root or os.path.dirname(path))
        target = os.path.join(output_root, os.path.splitext(relative)[0] + '.rpe.json')
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as handle:
            handle.write(produced_text)
        record['output'] = target

    result = None
    if verify:
        import tempfile
        handle = target
        temporary = None
        if handle is None:
            temporary = tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.json',
                                                    delete=False)
            temporary.write(produced_text)
            temporary.close()
            handle = temporary.name
        result = oracle.compare(path, handle)
        if temporary is not None:
            os.unlink(temporary.name)

    record.update({
        'ok': result['ok'] if result else None,
        'kind': 'pe',
        'lines': len(source.lines),
        'notes': sum(len(line.notes) for line in source.lines.values()),
        'offset': (source.offset_ms - 175.0) / 1000.0,
        'bpm': round(source.bpm[0][1], 6) if source.bpm else 120.0,
        'multi_bpm': len(source.bpm) > 1,
        'dmax': {
            'x': result['max']['x'], 'y': result['max']['y'],
            'angle': result['max']['angle'], 'alpha': result['max']['alpha'],
            'speed': result['max']['speed'], 'note_t': result['max']['note_t'],
            'note_x': result['max']['note_x'], 'hold': 0.0,
            'offset': result['offset_error'], 'hold_body': 0.0,
        } if result else {},
        'warnings': source.warnings[:12],
        'warning_count': len(source.warnings),
        'ignored': {},
        'issues': result['issues'][:6] if result else [],
        'rpe_bytes': len(produced_text),
        'ms': round((time.perf_counter() - started) * 1000, 1),
    })
    return record


def run_batch(args) -> int:
    records: list[dict[str, Any]] = []
    skipped = 0
    started = time.perf_counter()
    paths = list(_iter_json(args.input))
    for index, path in enumerate(paths, 1):
        record = convert_and_check(path, floor_policy=args.floor_policy, bpm=args.bpm,
                                   output_root=args.output, input_root=args.input,
                                   verify=not args.no_verify,
                                   independent=getattr(args, 'independent', False))
        if 'skipped' in record:
            skipped += 1
            continue
        records.append(record)
        if args.progress and (index % args.progress == 0 or index == len(paths)):
            print(f'  [{index}/{len(paths)}] {time.perf_counter() - started:.0f}s',
                  file=sys.stderr, flush=True)
        if args.limit and len(records) >= args.limit:
            break

    ok = [r for r in records if r.get('ok') is True]
    unverified = [r for r in records if r.get('ok') is None]
    failed = [r for r in records if r.get('ok') is False]
    elapsed = time.perf_counter() - started
    summary = f'converted {len(records)} chart(s): {len(ok)} exact, {len(failed)} failed'
    if unverified:
        summary += f', {len(unverified)} unverified'
    print(f'{summary} ({skipped} non-official file(s) skipped, {elapsed:.0f}s)')
    if ok:
        for channel in ('x', 'y', 'angle', 'alpha', 'speed', 'note_t', 'note_x',
                        'hold', 'offset', 'hold_body'):
            worst = max(ok, key=lambda r: r['dmax'][channel])
            print(f'  max {channel:<9} {worst["dmax"][channel]:.3e}   '
                  f'({os.path.basename(os.path.dirname(worst["path"]))}/'
                  f'{os.path.basename(worst["path"])})')
        multi = [r for r in ok if r['multi_bpm']]
        print(f'  charts with several line BPMs: {len(multi)}')
        print(f'  charts with warnings: {len({r["path"] for r in ok if r["warning_count"]})}')
        ignored: dict[str, int] = {}
        for record in ok:
            for level, keys in (record.get('ignored') or {}).items():
                group = re.sub(r'\[\d+\]$', '', level)
                for key in keys:
                    label = f'{group}.{key}'
                    ignored[label] = ignored.get(label, 0) + 1
        if ignored:
            print('  ignored non-convertible fields: ' +
                  ', '.join(f'{key} ({count})' for key, count in sorted(ignored.items())))
        holds = sorted(ok, key=lambda r: r['dmax']['hold_body'], reverse=True)
        affected = sum(1 for r in ok if r['dmax']['hold_body'] > 0.5)
        print(f'  hold-body deviation: max {holds[0]["dmax"]["hold_body"]:.3f} Y, '
              f'{affected} chart(s) above 0.5 Y (inherent model gap, see README)')
        print(f'  notes checked: {sum(r["notes"] for r in ok)}')
    for record in failed[:10]:
        print(f'  ! {record["path"]}: {record.get("error") or record.get("issues")}')

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(records, handle, ensure_ascii=False, indent=1)
        print(f'  report: {args.json_out}')
    return 0 if not failed else 1

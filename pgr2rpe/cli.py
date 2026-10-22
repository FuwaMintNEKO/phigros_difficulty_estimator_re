"""Command line interface: ``python -m pgr2rpe ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .model import Chart
from .official_in import load_official
from .official_out import write_official
from .pec_in import load_pec, pec_to_rpe
from .rpe_in import load_rpe
from .rpe_out import FLOOR_POLICY_AUTO, FLOOR_POLICY_DROP, FLOOR_POLICY_YOFFSET, write_rpe
from .spec import FormatError
from .verify import compare_charts


def detect_format(text: str) -> str:
    """Return ``'official'``, ``'rpe'`` or ``'pe'`` from content, never from the name."""
    stripped = text.lstrip('\ufeff \t\r\n')
    if not stripped.startswith('{'):
        # PEC is plain text: the first line is an integer millisecond offset.
        first = stripped.splitlines()[0].strip() if stripped else ''
        try:
            float(first)
        except ValueError as exc:
            raise FormatError('not a chart: first line is neither JSON nor a PEC offset '
                              f'(got {first[:40]!r})') from exc
        return 'pe'
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise FormatError(f'not valid JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise FormatError('chart root must be an object')
    if 'BPMList' in payload or 'META' in payload:
        return 'rpe'
    if 'judgeLineList' in payload:
        return 'official'
    raise FormatError('unrecognised chart: neither BPMList/META nor judgeLineList present')


def load_any(path: str):
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as handle:
        text = handle.read()
    kind = detect_format(text)
    if kind == 'official':
        return load_official(text, source=path), kind
    if kind == 'rpe':
        return load_rpe(text, source=path), kind
    return load_pec(text, source=path), kind


def _meta_from_args(args) -> dict[str, Any]:
    meta = {}
    for key in ('name', 'composer', 'charter', 'level', 'song', 'background', 'illustration', 'id'):
        value = getattr(args, key, None)
        if value is not None:
            meta[key] = value
    return meta


def _print_warnings(chart: Chart, limit: int = 8) -> None:
    if not chart.warnings:
        return
    print(f'  warnings: {len(chart.warnings)}')
    for message in chart.warnings[:limit]:
        print(f'    - {message}')
    if len(chart.warnings) > limit:
        print(f'    ... {len(chart.warnings) - limit} more')


def _write(args, payload) -> None:
    with open(args.output, 'w', encoding='utf-8') as handle:
        if args.pretty:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))


def cmd_convert(args) -> int:
    chart, kind = load_any(args.input)
    if kind == 'pe' and args.to == 'official':
        raise FormatError('PEC -> official is not supported: PEC uses 29 easing curves that '
                          'the official format cannot express. Convert to RPE first.')
    if kind == args.to:
        print(f'input is already a {kind} chart')

    if kind == 'pe':
        payload = pec_to_rpe(chart, meta=_meta_from_args(args),
                             rpe_version=args.rpe_version,
                             offset_bias_ms=args.pec_offset_bias)
        print(f'pe -> rpe: {args.input}')
        print(f'  lines: {len(chart.lines)}  notes: '
              f'{sum(len(line.notes) for line in chart.lines.values())}  '
              f'offset header: {chart.offset_ms:+g}ms')
        _write(args, payload)
        print(f'  wrote {os.path.getsize(args.output)} bytes to {args.output}')
        _print_warnings(chart)
        if args.verify and not _verify_output(args):
            return 2
        return 0

    if args.to == 'rpe':
        payload = write_rpe(chart,
                            bpm=args.bpm,
                            meta=_meta_from_args(args),
                            rpe_version=args.rpe_version,
                            floor_policy=args.floor_policy)
    else:
        payload = write_official(chart, format_version=args.format_version)

    _write(args, payload)

    print(f'{kind} -> {args.to}: {args.input}')
    print(f'  lines: {len(chart.lines)}  notes: {chart.note_count()}  '
          f'offset: {chart.offset:+.4f}s  duration: {chart.last_note_time:.3f}s')
    print(f'  wrote {os.path.getsize(args.output)} bytes to {args.output}')
    _print_warnings(chart)

    if args.verify:
        with open(args.output, 'r', encoding='utf-8') as handle:
            produced = handle.read()
        kind_out = detect_format(produced)
        back = (load_rpe if kind_out == 'rpe' else load_official)(produced, source=args.output)
        report = compare_charts(chart, back)
        print(f'  verify: {"OK" if report.ok else "FAILED"}  '
              f'(x {report.max_dx:.2e}, y {report.max_dy:.2e}, '
              f'angle {report.max_dangle:.2e}, alpha {report.max_dalpha:.2e}, '
              f'speed {report.max_dspeed:.2e}, note t {report.max_dtime:.2e}, '
              f'hold-body {report.hold_body_max_error:.3e} Y)')
        for issue in report.issues[:10]:
            print(f'    ! {issue}')
        if not report.ok:
            return 2
    return 0


def _verify_output(args) -> bool:
    """Check the written file against its source with the independent evaluator."""
    from . import oracle
    result = oracle.compare(args.input, args.output)
    print(f'  oracle: {"OK" if result["ok"] else "FAILED"}  '
          f'(x {result["max"]["x"]:.2e}, y {result["max"]["y"]:.2e}, '
          f'angle {result["max"]["angle"]:.2e}, alpha {result["max"]["alpha"]:.2e}, '
          f'speed {result["max"]["speed"]:.2e})')
    for issue in result['issues'][:10]:
        print(f'    ! {issue}')
    return result['ok']


def cmd_verify(args) -> int:
    """Compare two charts with the independent evaluator (any of the three formats)."""
    from . import oracle
    result = oracle.compare(args.left, args.right, samples=args.samples)
    print(f'{oracle.detect(args.left):>8} {args.left}')
    print(f'{oracle.detect(args.right):>8} {args.right}')
    print(json.dumps({k: v for k, v in result.items() if k != 'issues'},
                     ensure_ascii=False, indent=2, default=float))
    for issue in result['issues'][:20]:
        print(f'  ! {issue}')
    return 0 if result['ok'] else 1


def cmd_batch(args) -> int:
    from .batch import run_batch
    return run_batch(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='pgr2rpe',
        description='Convert Phigros charts (official JSON or PEC text) to '
                    'Re:PhiEdit (RPE) format and prove the conversion is exact.')
    sub = parser.add_subparsers(dest='command', required=True)

    convert = sub.add_parser('convert', help='convert one chart')
    convert.add_argument('input')
    convert.add_argument('-o', '--output', required=True)
    convert.add_argument('--to', choices=['rpe', 'official'], default='rpe',
                         help='target format (PEC can only become RPE)')
    convert.add_argument('--bpm', type=float, default=None,
                         help='global BPM for the RPE beat axis (default: the chart\'s own)')
    convert.add_argument('--rpe-version', type=int, default=170)
    convert.add_argument('--format-version', type=int, choices=[1, 3], default=3,
                         help='output formatVersion when --to official')
    convert.add_argument('--floor-policy', choices=[FLOOR_POLICY_AUTO, FLOOR_POLICY_DROP,
                                                   FLOOR_POLICY_YOFFSET],
                         default=FLOOR_POLICY_AUTO,
                         help='how to handle notes whose floorPosition is off the line')
    convert.add_argument('--pec-offset-bias', type=float, default=175.0,
                         help='PEC first-line bias in ms (docs: 175, Phira uses 150)')
    convert.add_argument('--no-verify', dest='verify', action='store_false')
    convert.add_argument('--pretty', action='store_true')
    for key in ('name', 'composer', 'charter', 'level', 'song', 'background',
                'illustration', 'id'):
        convert.add_argument(f'--{key}', default=None)
    convert.set_defaults(func=cmd_convert)

    verify = sub.add_parser('verify',
                            help='compare two charts with the independent evaluator')
    verify.add_argument('left')
    verify.add_argument('right')
    verify.add_argument('--samples', type=int, default=1200)
    verify.set_defaults(func=cmd_verify)

    batch = sub.add_parser('batch', help='convert every chart under a directory')
    batch.add_argument('input')
    batch.add_argument('-o', '--output', default=None,
                       help='write converted .rpe.json files under this directory')
    batch.add_argument('--limit', type=int, default=0)
    batch.add_argument('--floor-policy', choices=[FLOOR_POLICY_AUTO, FLOOR_POLICY_DROP,
                                                 FLOOR_POLICY_YOFFSET],
                       default=FLOOR_POLICY_AUTO)
    batch.add_argument('--bpm', type=float, default=None)
    batch.add_argument('--json-out', default=None)
    batch.add_argument('--progress', type=int, default=0, metavar='N',
                       help='print progress to stderr every N files')
    batch.add_argument('--no-verify', dest='no_verify', action='store_true',
                       help='bulk export only: skip the semantic self-check (much faster)')
    batch.add_argument('--independent', action='store_true',
                       help='verify with pgr2rpe.oracle instead of the in-model comparator')
    batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FormatError, ValueError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

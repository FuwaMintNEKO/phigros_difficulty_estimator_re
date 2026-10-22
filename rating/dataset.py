"""Dataset assembly: official charts, labelled community charts, test folder."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import os
import pickle
import time

import numpy as np

from . import scale
from .charts import ChartData, load_chart
from .features import FEATURE_VERSION, extract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training / evaluation data is not shipped with the repository (the charts are
# not ours to redistribute). Point PHIGROS_DATA_DIR at a folder laid out as:
#   chart/<Song.Artist.0>/{EZ,HD,IN,AT}.json   official charts
#   info/difficulty.tsv                        official constants
#   phira/json_unranked_4star/<id>.json        labelled community charts
#   phira/v130d_manifest17plus_predictions.csv their labels and train/holdout split
#   phira/json/<id>.json + phira/ranked_compare.csv   reference-only set
#   test_charts/                               charts to predict
DATA_ROOT = os.path.abspath(os.environ.get('PHIGROS_DATA_DIR') or os.path.join(ROOT, 'data'))
OFFICIAL_DIR = os.path.join(DATA_ROOT, 'chart')
OFFICIAL_TSV = os.path.join(DATA_ROOT, 'info', 'difficulty.tsv')
CUSTOM_DIR = os.path.join(DATA_ROOT, 'phira', 'json_unranked_4star')
CUSTOM_CSV = os.path.join(DATA_ROOT, 'phira', 'v130d_manifest17plus_predictions.csv')
TEST_DIR = os.path.join(DATA_ROOT, 'test_charts')
RANKED_DIR = os.path.join(DATA_ROOT, 'phira', 'json')
RANKED_CSV = os.path.join(DATA_ROOT, 'phira', 'ranked_compare.csv')

LEVELS = ('EZ', 'HD', 'IN', 'AT')


@dataclass
class Record:
    path: str
    source: str            # official / community / test
    level: str = ''
    song: str = ''
    label: float = float('nan')          # official scale
    label_community: float = float('nan')
    role: str = ''
    name: str = ''


def official_records() -> list[Record]:
    labels: dict[str, dict[str, float]] = {}
    with open(OFFICIAL_TSV, 'r', encoding='utf-8') as handle:
        for row in csv.reader(handle, delimiter='\t'):
            if len(row) < 2:
                continue
            labels[row[0].strip()] = {
                level: float(value) for level, value in zip(LEVELS, row[1:])
                if value.strip()
            }
    records = []
    for folder in sorted(os.listdir(OFFICIAL_DIR)):
        folder_path = os.path.join(OFFICIAL_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        song = folder[:-2] if folder.endswith('.0') else folder
        table = labels.get(song)
        if not table:
            continue
        for level in LEVELS:
            path = os.path.join(folder_path, f'{level}.json')
            if not os.path.exists(path) or level not in table:
                continue
            records.append(Record(path=path, source='official', level=level, song=song,
                                  label=table[level],
                                  label_community=float(scale.official_to_community(table[level]))))
    return records


def custom_records() -> list[Record]:
    records = []
    with open(CUSTOM_CSV, 'r', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            path = os.path.join(CUSTOM_DIR, row['id'] + '.json')
            if not os.path.exists(path):
                continue
            value = (row.get('diff(高手表)') or '').strip()
            if not value:
                continue
            community = float(value)
            records.append(Record(path=path, source='community', song=row['id'],
                                  name=row.get('name', ''),
                                  label=float(scale.community_to_official(community)),
                                  label_community=community,
                                  role=(row.get('role') or '').strip()))
    return records


def ranked_records() -> list[Record]:
    """Phira "ranked" charts: community-rated, same scale as the 高手表.

    ``ranked_compare.csv`` carries the phira ranking for 558 charts from 8.5 to
    18.5. They are the only custom charts that cover the 14-17 band, which is
    exactly where the model used to over-predict (dense RPE charts were being
    read as 17+ because the official training set has no such charts).
    """
    records = []
    with open(RANKED_CSV, 'r', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            path = os.path.join(RANKED_DIR, row['id'] + '.json')
            value = (row.get('diff') or '').strip()
            if not value or not os.path.exists(path):
                continue
            community = float(value)
            records.append(Record(path=path, source='ranked', song=row['id'],
                                  name=row.get('name', ''),
                                  label=float(scale.community_to_official(community)),
                                  label_community=community))
    return records


def test_records() -> list[Record]:
    records = []
    for name in sorted(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, name)
        if os.path.isdir(path):
            continue
        records.append(Record(path=path, source='test', name=name))
    return records


def _cache_key(path: str) -> str:
    stat = os.stat(path)
    return f'{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}|{FEATURE_VERSION}'


def build_matrix(records: list[Record], cache_path: str | None = None,
                 verbose: bool = True) -> tuple[list[dict], np.ndarray, list[str]]:
    """Extract features for every record, with a path+mtime keyed cache."""
    cache: dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            cache = pickle.load(handle)

    rows, failed = [], []
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        key = _cache_key(record.path)
        features = cache.get(key)
        if features is None:
            try:
                features = extract(load_chart(record.path))
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed.append((record.path, f'{type(exc).__name__}: {exc}'))
                continue
            cache[key] = features
        rows.append({'record': record, 'features': features})
        if verbose and index % 200 == 0:
            print(f'  features {index}/{len(records)}  {time.perf_counter()-started:.0f}s',
                  flush=True)

    if cache_path:
        with open(cache_path, 'wb') as handle:
            pickle.dump(cache, handle)

    names = sorted({name for row in rows for name in row['features']})
    matrix = np.asarray([[row['features'].get(name, 0.0) for name in names] for row in rows],
                        dtype=float)
    if failed and verbose:
        print(f'  {len(failed)} chart(s) failed to parse')
        for path, why in failed[:8]:
            print(f'    ! {os.path.basename(path)}: {why}')
    return rows, matrix, names

"""Build (and cache) the feature matrix for official + community + test charts."""

from __future__ import annotations

import os
import pickle
import sys

from .dataset import (ROOT, build_matrix, custom_records, official_records,
                      ranked_records, test_records)

CACHE = os.path.join(ROOT, 'outputs', 'feature_cache.pkl')
DATASET = os.path.join(ROOT, 'outputs', 'dataset.pkl')


def main() -> int:
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    official = official_records()
    community = custom_records()
    ranked = ranked_records()
    tests = test_records()
    print(f'official {len(official)}  ranked {len(ranked)}  community {len(community)}  '
          f'test {len(tests)}')
    rows, matrix, names = build_matrix(official + ranked + community + tests, cache_path=CACHE)
    with open(DATASET, 'wb') as handle:
        pickle.dump({'rows': rows, 'matrix': matrix, 'names': names}, handle)
    print(f'features: {len(names)} dims, {len(rows)} charts -> {DATASET}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

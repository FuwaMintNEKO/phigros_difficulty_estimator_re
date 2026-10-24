"""Difficulty-scale alignment: community (高手表) scale <-> official constants.

The two scales coincide at the bottom and diverge above: a community chart rated
17.6 is worth 17.6 in official terms, 17.9 community is about 17.9 official, and
18.0 official already corresponds to ~18.8 on the community scale - the official
ladder compresses the top end, so community ratings spread out much faster.

Anchors used (given by the project owner):

=================  =================
official           community
=================  =================
17.6               17.6
17.9               18.4
18.0               18.8
=================  =================

Below 17.6 the two scales are taken as identical (the community table is
calibrated against the official constants there). The curve is a monotone cubic
Hermite spline (PCHIP): it passes exactly through every anchor, has a continuous
first derivative, never overshoots, and needs no thresholds - so a 0.01 change in
a community label always produces a 0.01-scale change in the training target.
"""

from __future__ import annotations

import numpy as np
from .pchip import Pchip

# (official, community)
ANCHORS = [
    (12.0, 12.0),
    (14.0, 14.0),
    (15.0, 15.0),
    (16.0, 16.0),
    (16.5, 16.5),
    (17.0, 17.0),
    (17.2, 17.2),
    (17.4, 17.4),
    (17.6, 17.6),
    (17.9, 18.2),
    (18.0, 18.6),
]

_OFFICIAL = np.array([a[0] for a in ANCHORS], dtype=float)
_COMMUNITY = np.array([a[1] for a in ANCHORS], dtype=float)

# The spline is only trusted inside the anchored range; both ends continue
# linearly with the boundary slope (which is 1.0 below 17.6 and 4.0 above 18.0),
# so the map never overshoots or folds back the way a free cubic extrapolation
# would.
_official_to_community_spline = Pchip(_OFFICIAL, _COMMUNITY)
_LAST_SLOPE = (_COMMUNITY[-1] - _COMMUNITY[-2]) / (_OFFICIAL[-1] - _OFFICIAL[-2])
_FIRST_SLOPE = (_COMMUNITY[1] - _COMMUNITY[0]) / (_OFFICIAL[1] - _OFFICIAL[0])


def official_to_community(value):
    """Map an official constant to the community (高手表) scale."""
    x = np.asarray(value, dtype=float)
    inside = _official_to_community_spline(np.clip(x, _OFFICIAL[0], _OFFICIAL[-1]))
    below = _COMMUNITY[0] + (x - _OFFICIAL[0]) * _FIRST_SLOPE
    above = _COMMUNITY[-1] + (x - _OFFICIAL[-1]) * _LAST_SLOPE
    return np.where(x < _OFFICIAL[0], below, np.where(x > _OFFICIAL[-1], above, inside))


def community_to_official(value):
    """Map a community (高手表) rating to the official constant scale."""
    y = np.asarray(value, dtype=float)
    low = np.full_like(y, _OFFICIAL[0] - 6.0)
    high = np.full_like(y, _OFFICIAL[-1])
    for _ in range(80):
        mid = 0.5 * (low + high)
        below = official_to_community(mid) < y
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    inside = 0.5 * (low + high)
    # 两端之外正映射是线性的，直接解析求逆；二分下界只保证覆盖锚点区间，
    # 用它兜底会把 6.0 以下的官方定数（EZ 谱低至 0.5）全部夹到 6.0。
    above = _OFFICIAL[-1] + (y - _COMMUNITY[-1]) / _LAST_SLOPE
    below = _OFFICIAL[0] + (y - _COMMUNITY[0]) / _FIRST_SLOPE
    return np.where(y > _COMMUNITY[-1], above, np.where(y < _COMMUNITY[0], below, inside))


def describe() -> dict:
    probe = [12, 14, 15, 16, 17, 17.3, 17.6, 17.75, 17.9, 18.0]
    return {
        'anchors': ANCHORS,
        'official_to_community': {x: round(float(official_to_community(x)), 3) for x in probe},
        'community_to_official': {
            x: round(float(community_to_official(x)), 3)
            for x in (17.6, 17.9, 18.0, 18.4, 18.8, 19.0, 19.3)},
    }


if __name__ == '__main__':
    import json
    print(json.dumps(describe(), ensure_ascii=False, indent=2))

"""One place that turns a chart file into a difficulty number."""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import replace

import numpy as np

from scipy.interpolate import PchipInterpolator

from . import scale
from .charts import load_chart
from .compose import CATEGORIES, decomposition
from .features import extract
from .model import MODEL_PATH

class Predictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'找不到模型文件 {model_path}。请先训练（python -m rating.train）或把 model.pkl 放到 models/ 下。')
        with open(model_path, 'rb') as handle:
            self.bundle = pickle.load(handle)
        self.columns = self.bundle['columns']
        self.baseline = self.bundle['baseline']
        self._cache: dict[str, object] = {}
        self._extra: dict[str, dict] = {}
        self._curves: dict[str, tuple] = {}

    # ---- scoring -------------------------------------------------------
    def vector(self, features: dict) -> np.ndarray:
        return np.asarray([[features.get(name, 0.0) for name in self.columns]])

    def score(self, vector: np.ndarray) -> float:
        b = self.bundle
        low, high = b.get('blend', (17.1, 17.9))
        base = float(b['calib_full'](b['full'].predict(vector)[0]))
        weight = min(1.0, max(0.0, (base - low) / (high - low)))
        raw_top = float(b['calib_top'](b['top'].predict(vector)[0]))
        tail = b.get('tail_top')
        if tail is not None:
            raw_top = float(tail(vector, np.asarray([raw_top]))[0])
            weight = max(weight, min(1.0, float(tail.excess(vector)[0]) / 0.35))
        value = (1 - weight) * base + weight * float(scale.community_to_official(raw_top))
        if tail is not None:
            value += float(tail.official_bonus(vector)[0])
        if b.get('bias_correction') is not None:
            value = float(b['bias_correction'](np.asarray([value]))[0])
        if b.get('high_lift') is not None:
            value = float(b['high_lift'](value))
        return value

    # ---- per chart -----------------------------------------------------
    def chart(self, path: str):
        """Parsed chart, cached by content hash (the browser re-uploads it)."""
        digest = hashlib.md5(open(path, 'rb').read()).hexdigest()
        if digest not in self._cache:
            chart = load_chart(path)
            if chart.n_notes < 8 or chart.duration <= 1.0:
                raise ValueError(f'只解析到 {chart.n_notes} 个音符，这张谱可能不是有效谱面')
            self._cache[digest] = chart
        return self._cache[digest]

    def features(self, path: str) -> dict:
        chart = self.chart(path)
        key = 'features'
        if key not in self._extra:
            self._extra[key] = extract(chart)
        return self._extra[key]

    def scaled_features(self, path: str, speed: float) -> dict:
        """Features of the chart as it plays at ``speed``.

        Phira's speed control rescales BPM, note spacing, judge-line animation and
        scroll velocity together - the whole chart plays k times faster, exactly
        like a sped-up video. So the honest way to predict it is to rebuild the
        chart with every time divided by k (and every velocity multiplied), then
        extract features again; that also gets the fixed-window quantities
        (chords within 50 ms, the last-20-second finale) right, which a per-feature
        rescale cannot.
        """
        chart = self.chart(path)
        if abs(speed - 1.0) < 1e-9:
            return extract(chart)
        squeezed = replace(chart, times=chart.times / speed,
                           hold_ends=chart.hold_ends / speed,
                           duration=chart.duration / speed,
                           speeds=chart.speeds * speed)
        return extract(squeezed)

    def speed_curve(self, path: str, low: float = 0.6, high: float = 2.0,
                    points: int = 15):
        """Difficulty as a function of speed, made monotone.

        A faster chart is never easier, but the model is built from discrete
        counts and piecewise-constant trees, so its raw curve has small dips and
        one visible cliff: on Archidoxen the score jumps 0.20 at 1.87x because a
        whole cluster of notes falls inside the 50 ms chord window at once. The
        curve is therefore sampled and made non-decreasing (running maximum)
        before being interpolated; the displayed value always follows it.
        """
        digest = hashlib.md5(open(path, 'rb').read()).hexdigest()
        cached = self._curves.get(digest)
        if cached is not None:
            return cached
        speeds = np.linspace(low, high, points)
        scores = np.asarray([self.explain(path, float(sp))['official'] for sp in speeds])
        scores = np.maximum.accumulate(scores)
        self._curves[digest] = (speeds, scores)
        return speeds, scores

    def explain(self, path: str, speed: float = 1.0, monotone: bool = False) -> dict:
        raw = extract(self.chart(path))
        features = self.scaled_features(path, speed)
        result = decomposition(self.score, features, self.baseline, self.columns)
        official = result['score']
        chart = self.chart(path)
        kind = chart.kind
        # reported at the speed being previewed: at 1.3x the chart really is
        # 1.3x denser and 1.3x shorter
        stats = {
            'notes': int(features.get('n_notes', 0)),
            'duration': round(float(features.get('duration', 0.0)), 1),
            'press_nps': round(float(features.get('press_nps', 0.0)), 2),
            'peak': round(float(features.get('d0.5_max', 0.0)), 1),
            'finale': round(float(features.get('finale_d0.5_max', 0.0)), 1),
            'hold_ratio': round(float(features.get('hold_frac', 0.0)), 3),
            'mix': {
                'tap': round(float(features.get('frac_tap', 0.0)), 4),
                'hold': round(float(features.get('hold_frac', 0.0)), 4),
                'drag': round(float(features.get('frac_drag', 0.0)), 4),
                'flick': round(float(features.get('frac_flick', 0.0)), 4),
            },
            'drag_ratio': round(float(features.get('frac_drag', 0.0)), 3),
            'base_duration': round(float(raw.get('duration', 0.0)), 1),
        }
        if monotone and abs(speed - 1.0) > 1e-9:
            speeds, scores = self.speed_curve(path, points=9)
            official = float(np.interp(speed, speeds, scores))

        return {
            'name': os.path.basename(os.path.dirname(path)) or os.path.basename(path),
            'file': os.path.basename(path),
            'kind': kind,
            'official': official,
            'community': float(scale.official_to_community(official)),
            'raw_official': result['score'],
            'percent': result['percent'],
            'contributions': result['contributions'],
            'stats': stats,
            'speed': speed,
            'meta': {
                'title': (chart.meta or {}).get('name', ''),
                'charter': (chart.meta or {}).get('charter', ''),
                'composer': (chart.meta or {}).get('composer', ''),
                'level': (chart.meta or {}).get('level', ''),
                'bpm_min': (chart.meta or {}).get('bpm_min'),
                'bpm_max': (chart.meta or {}).get('bpm_max'),
            },
            'categories': {k: spec['label'] for k, spec in CATEGORIES.items()},
            'hints': {k: spec['hint'] for k, spec in CATEGORIES.items()},
        }

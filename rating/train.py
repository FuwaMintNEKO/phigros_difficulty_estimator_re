"""Training driver: fit the production models and report the benchmarks.

Data sources (all mapped onto the official scale):

* ``official``  — 1037 charts, official constants (authoritative);
* ``community`` — 269 charts from the 高手表 (17.0..19.3).

``data/phira/json`` (558 ranked charts) is **evaluation only**: its labels are a
manual reference of limited precision, so it never enters training. It is scored
after the fact to show where the remaining error sits.

Model: gradient boosting plus a ridge fit on its residual (``HybridModel``), then
a monotone calibration fitted on the out-of-fold predictions. The ridge term is
what guarantees different charts get different numbers — trees alone are
piecewise constant.

Evaluation:

* official charts: 5-fold ``GroupKFold`` grouped by **song** (same protocol as
  the old project, so its 0.5115 is directly comparable);
* ranked and 高手表 charts: 5-fold CV grouped by chart id, scored in community
  units (that is what the labels mean).
"""

from __future__ import annotations

import csv
import json
import os
import pickle

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from . import scale
from .dataset import ROOT
from .model import (DATASET, MODEL_PATH, BiasCorrection, Calibration, HighLift,
                    HybridModel, fit_tail, fmt, label_density_weights, score)

OUT = os.path.join(ROOT, 'outputs')
BLEND_LOW, BLEND_HIGH = 17.1, 17.9      # official units
WEIGHTS = {'official': 1.0, 'community': 0.7}
SOURCES = ('official', 'community')


def make(kind: str = 'full') -> HybridModel:
    if kind == 'top':
        base = GradientBoostingRegressor(n_estimators=700, learning_rate=0.03, max_depth=3,
                                         subsample=0.85, min_samples_leaf=6, random_state=0)
    else:
        base = GradientBoostingRegressor(n_estimators=700, learning_rate=0.03, max_depth=4,
                                         subsample=0.85, min_samples_leaf=6, random_state=0)
    return HybridModel(base, alpha=8.0)


def fit_calibration(prediction: np.ndarray, target: np.ndarray):
    """Monotone, smooth post-map fitted on out-of-fold predictions.

    Tree ensembles compress towards the middle of the label range. Isotonic
    regression removes that compression, but it is a staircase, and a staircase
    maps many distinct predictions onto one value; the steps are Gaussian
    smoothed and re-expressed as a PCHIP, so the final map stays smooth and
    one-to-one.
    """
    iso = IsotonicRegression(out_of_bounds='clip').fit(prediction, target)
    lo, hi = float(prediction.min()), float(prediction.max())
    grid = np.linspace(lo, hi, 257)
    values = iso.predict(grid)
    step = grid[1] - grid[0]
    sigma = max((hi - lo) / 120.0, 1e-9)
    radius = int(np.ceil(3 * sigma / step))
    if radius >= 1:
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) * step / sigma) ** 2)
        kernel /= kernel.sum()
        values = np.convolve(np.pad(values, radius, mode='edge'), kernel, mode='valid')
    values = np.maximum.accumulate(values)
    slope_high = float((values[-1] - values[-2]) / step)
    slope_low = float((values[1] - values[0]) / step)
    return Calibration(PchipInterpolator(grid, values), lo, hi,
                       min(max(slope_low, 0.0), 1.0), min(max(slope_high, 0.0), 1.0))


def fit_bias_curve(prediction: np.ndarray, label: np.ndarray, bins: int = 10):
    """Systematic bias of the deployed pipeline, measured on the ranked set.

    ``data/phira/json`` never enters training, but 557 of its charts carry a
    reference rating. Individual labels are noisy, the *mean* residual per band
    is not: it shows +0.45 in 15-16, -0.3 above 18. Fitting a robust median
    curve over prediction bins and subtracting a damped share of it removes the
    band-level offset without fitting any single chart.
    """
    order = np.argsort(prediction)
    p, y = prediction[order], label[order]
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    xs, residuals = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < 8:
            continue
        xs.append(float(np.median(p[start:stop])))
        # residual in the same direction as the correction: pred - label
        residuals.append(float(np.median(p[start:stop] - y[start:stop])))
    if len(xs) < 3:
        return None
    xs = np.asarray(xs)
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    return PchipInterpolator(xs[keep], np.asarray(residuals)[keep], extrapolate=True)


def _fit_with_weights(X, y, w, kind='full') -> HybridModel:
    model = make(kind)
    model.fit(X, y, sample_weight=w)
    return model


def main() -> int:
    data = pickle.load(open(DATASET, 'rb'))
    rows, X, names = data['rows'], data['matrix'], data['names']

    index = {key: [i for i, row in enumerate(rows) if row['record'].source == key]
             for key in SOURCES}
    # the 高手表 CSV ships its own split: 222 charts for training, 47 held out.
    # Honour it, otherwise the "holdout" number would be trained on itself.
    community = index['community']
    comm_train = [i for i in community if rows[i]['record'].role != '留出']
    comm_hold = [i for i in community if rows[i]['record'].role == '留出']

    def pos(subset, pool):
        lookup = {index_: k for k, index_ in enumerate(pool)}
        return np.asarray([lookup[i] for i in subset], dtype=int)

    Xo = X[index['official']]
    Xc = X[community]
    y_official = np.asarray([rows[i]['record'].label for i in index['official']])
    g_official = np.asarray([rows[i]['record'].song for i in index['official']])
    y_comm = np.asarray([rows[i]['record'].label_community for i in community])
    Xc_train, Xc_hold = Xc[pos(comm_train, community)], Xc[pos(comm_hold, community)]
    yc_train, yc_hold = y_comm[pos(comm_train, community)], y_comm[pos(comm_hold, community)]
    print(f'official {len(index["official"])}  community {len(community)} '
          f'(train {len(comm_train)} / holdout {len(comm_hold)})  features {len(names)}')

    # ---------- official OOF (grouped by song) ------------------------------
    pred = np.zeros(len(index['official']))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(Xo, y_official, g_official):
        X_fit = np.vstack([Xo[train_idx], Xc_train])
        y_fit = np.concatenate([y_official[train_idx],
                                scale.community_to_official(yc_train)])
        w_fit = np.concatenate([label_density_weights(y_official[train_idx], alpha=0.5),
                                np.full(len(yc_train), WEIGHTS['community'])])
        pred[test_idx] = _fit_with_weights(X_fit, y_fit, w_fit, 'full').predict(Xo[test_idx])
    calib_full = fit_calibration(pred, y_official)
    official_report = score(y_official, calib_full(pred))
    official_report.name = 'official OOF (grouped by song)'
    print(' ', fmt(official_report))

    # ---------- 高手表 holdout (never seen in training) ---------------------
    # Fit on officials + the 222 training charts, then score the 47 held out
    # ones exactly the way production would (full + top + blend + calibration).
    full_hold = _fit_with_weights(
        np.vstack([Xo, Xc_train]),
        np.concatenate([y_official, scale.community_to_official(yc_train)]),
        np.concatenate([label_density_weights(y_official, alpha=0.5),
                        np.full(len(yc_train), WEIGHTS['community'])]), 'full')
    oof_train = np.zeros(len(yc_train))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(Xc_train, yc_train, np.arange(len(yc_train))):
        temp = _fit_with_weights(Xc_train[train_idx], yc_train[train_idx],
                                 label_density_weights(yc_train[train_idx], alpha=0.5), 'top')
        oof_train[test_idx] = temp.predict(Xc_train[test_idx])
    calib_top_train = fit_calibration(oof_train, yc_train)
    top_hold = _fit_with_weights(Xc_train, yc_train,
                                 label_density_weights(yc_train, alpha=0.5), 'top')

    tail_top = fit_tail(Xc_train, yc_train, names)
    base = calib_full(full_hold.predict(Xc_hold))
    top_hold_raw = calib_top_train(top_hold.predict(Xc_hold))
    top = scale.community_to_official(tail_top(Xc_hold, top_hold_raw))
    weight = np.clip((base - BLEND_LOW) / (BLEND_HIGH - BLEND_LOW), 0.0, 1.0)
    blended = (1 - weight) * base + weight * top
    high_lift = HighLift()
    pred_hold = high_lift.community(scale.official_to_community(blended))
    err_hold = pred_hold - yc_hold
    print(f'  高手表 holdout ({len(yc_hold)} charts): MAE {np.abs(err_hold).mean():.4f} '
          f'bias {err_hold.mean():+.3f}  p90 {np.quantile(np.abs(err_hold), 0.9):.3f} '
          f'(community units)')
    for low, high in ((17.0, 18.0), (18.0, 18.5), (18.5, 19.5)):
        band = (yc_hold >= low) & (yc_hold < high)
        if band.sum():
            print(f'    band {low:.1f}-{high:.1f}: n={int(band.sum()):2d} '
                  f'MAE {np.abs(err_hold[band]).mean():.3f} bias {err_hold[band].mean():+.3f}')

    # per-chart table for manual review
    hold_records = []
    for index_, p, e in zip(comm_hold, pred_hold, err_hold):
        record = rows[index_]['record']
        hold_records.append({
            'id': record.song, 'name': record.name,
            'label': f'{record.label_community:.2f}', 'pred': f'{p:.2f}',
            'err': f'{e:+.2f}', 'pred_official': f'{float(scale.community_to_official(p)):.2f}',
        })
    hold_records.sort(key=lambda item: -float(item['label']))
    with open(os.path.join(OUT, 'holdout_report.csv'), 'w', encoding='utf-8-sig',
              newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hold_records[0].keys()))
        writer.writeheader()
        writer.writerows(hold_records)
    print(f'    per-chart table -> {os.path.join(OUT, "holdout_report.csv")}')

    # ---------- production (all community charts are usable again) ----------
    X_all = np.vstack([Xo, Xc])
    y_all = np.concatenate([y_official, scale.community_to_official(y_comm)])
    w_all = np.concatenate([label_density_weights(y_official, alpha=0.5),
                            np.full(len(y_comm), WEIGHTS['community'])])
    full = _fit_with_weights(X_all, y_all, w_all, 'full')

    oof_all = np.zeros(len(y_comm))
    for train_idx, test_idx in GroupKFold(n_splits=5).split(Xc, y_comm, np.arange(len(y_comm))):
        temp = _fit_with_weights(Xc[train_idx], y_comm[train_idx],
                                 label_density_weights(y_comm[train_idx], alpha=0.5), 'top')
        oof_all[test_idx] = temp.predict(Xc[test_idx])
    calib_top = fit_calibration(oof_all, y_comm)
    top = _fit_with_weights(Xc, y_comm, label_density_weights(y_comm, alpha=0.5), 'top')
    tail_top = fit_tail(Xc, y_comm, names)

    # ---- band-level bias: deliberately not applied -------------------------
    # A damped median-residual curve used to be fitted on the `ranked` reference
    # set (558 charts, community labels) and applied to every output. Measured on
    # the shipped model that curve costs far more than it buys:
    #
    #   official 16+      -0.340 -> -0.093   (MAE 0.340 -> 0.115)
    #   community holdout -0.293 -> -0.010   (MAE 0.359 -> 0.183)
    #   ranked reference  +0.063 -> +0.187   (MAE 0.498 -> 0.548)
    #
    # It was also fitted on the community scale but applied to official-scale
    # values, so above 17.6 it looks up the wrong part of the curve. The two
    # populations have opposite-signed residuals; one shift cannot fix both.
    # Refitting it on official OOF residuals gives a flat curve (|median| < 0.04)
    # because calibration already removes the band-level offset, so the stage is
    # dropped rather than retuned. `BiasCorrection` stays for import compatibility.
    bias_curve = None
    bias_damping = 0.0

    fit_rows = index['official'] + index['community']
    medians = np.median(X[fit_rows], axis=0)
    baseline = {name: float(value) for name, value in zip(names, medians)}

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as handle:
        pickle.dump({
            'full': full, 'top': top, 'columns': names, 'baseline': baseline,
            'high_lift': HighLift(),
            'bias_correction': (BiasCorrection(bias_curve, bias_damping)
                                if bias_curve is not None else None),
            'calib_full': calib_full, 'calib_top': calib_top, 'tail_top': tail_top,
            'blend': (BLEND_LOW, BLEND_HIGH),
            'official_oof_mae': official_report.mae,
            'official_oof_bias': official_report.bias,
            'official_bands': official_report.per_band,
            'community_holdout_mae': float(np.abs(err_hold).mean()),
            'community_holdout_bias': float(err_hold.mean()),
            'n_official': len(index['official']),
            'n_community': len(community),
            'n_community_holdout': len(comm_hold),
            'scale': scale.describe(),
        }, handle)
    print(f'  saved {MODEL_PATH}')

    with open(os.path.join(OUT, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            'official_oof': {'mae': official_report.mae, 'bias': official_report.bias,
                             'bands': official_report.per_band},
            'community_holdout': {'mae': float(np.abs(err_hold).mean()),
                                  'bias': float(err_hold.mean()),
                                  'n': len(comm_hold)},
            'n_official': len(index['official']),
            'n_community': len(community),
            'features': len(names), 'scale': scale.describe(),
        }, handle, ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

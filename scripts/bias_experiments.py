"""Why the first global model under-forecast, and what fixed it.

Fits each candidate on a random 1,500-series subset at two origins (the first
backtest fold and the M5 public-leaderboard window) and reports total bias and error.
Run from the project root:  python scripts/bias_experiments.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting import data, features, models  # noqa: E402
from demand_forecasting.config import SEED  # noqa: E402
from demand_forecasting.data import M5Panel  # noqa: E402

BASE = dict(learning_rate=0.08, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
            bagging_freq=1, lambda_l2=0.1, max_bin=127, seed=SEED, verbose=-1, num_threads=os.cpu_count())


def main():
    p = data.load()
    idx = np.sort(np.random.default_rng(0).choice(len(p.sales), 1500, replace=False))
    sp = M5Panel(p.ids.iloc[idx].reset_index(drop=True), p.sales[idx], p.price[idx], p.calendar)
    fb = features.FeatureBuilder(sp)
    for o in (1857, 1913):
        y = sp.sales[:, o:o + 28]
        print(f"origin d_{o}: 28-day moving average total bias {100 * (models.moving_average(sp, o).sum() - y.sum()) / y.sum():+.1f}%")
        for name, horizon_aware, objective, scale_free in [
            ("all features lagged 28+ days (tweedie)", False, "tweedie", False),
            ("horizon-aware (tweedie)", True, "tweedie", False),
            ("horizon-aware (poisson)", True, "poisson", False),
            ("horizon-aware + scale-free target", True, "regression", True),
        ]:
            t = time.time()
            tr = fb.build(np.arange(o - 730, o), random_horizon_seed=SEED if horizon_aware else None)
            te = fb.build(np.arange(o, o + 28), with_target=False, origin=o if horizon_aware else None)
            cols = features.feature_columns(tr)
            params = {**BASE, "objective": objective}
            if objective == "tweedie":
                params["tweedie_variance_power"] = 1.1
            lvl_tr = tr.rmean_28.fillna(0).to_numpy() + 0.5 if scale_free else 1.0
            lvl_te = te.rmean_28.fillna(0).to_numpy() + 0.5 if scale_free else 1.0
            ds = lgb.Dataset(tr[cols], tr.y / lvl_tr, weight=lvl_tr if scale_free else None,
                             categorical_feature=features.CATEGORICAL)
            m = lgb.train(params, ds, 300)
            pred = np.zeros_like(y)
            pred[te.series, te.day - o] = np.clip(m.predict(te[cols]) * lvl_te, 0, None)
            bias = 100 * (pred.sum() - y.sum()) / y.sum()
            total_rmse = np.sqrt(((pred.sum(0) - y.sum(0)) ** 2).mean())
            print(f"  {name:42s} total bias {bias:+6.1f}%  daily-total RMSE {total_rmse:6.1f}  item MAE {np.abs(pred - y).mean():.3f}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

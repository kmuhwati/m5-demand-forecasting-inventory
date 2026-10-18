"""Forecasters. Each returns a (n_series, HORIZON) matrix for one origin.

Baselines are deliberately strong and cheap: in M5 many teams failed to beat
a well-chosen exponential-smoothing or moving-average benchmark.
"""
from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from .config import HORIZON, SEED, TRAIN_DAYS
from .data import M5Panel
from .features import CATEGORICAL, FeatureBuilder, feature_columns

# --------------------------------------------------------------------------- baselines


def seasonal_naive(p: M5Panel, origin: int) -> np.ndarray:
    """Repeat the last observed week."""
    last_week = p.sales[:, origin - 7:origin]
    return np.tile(last_week, HORIZON // 7)


def moving_average(p: M5Panel, origin: int, window: int = 28) -> np.ndarray:
    hist = p.sales[:, origin - window:origin]
    shelf = p.on_shelf()[:, origin - window:origin]
    mean = np.where(shelf, hist, 0).sum(1) / np.maximum(shelf.sum(1), 1)
    return np.repeat(mean[:, None], HORIZON, axis=1)


def weekday_profile_ma(p: M5Panel, origin: int, weeks: int = 8) -> np.ndarray:
    """Moving average level x day-of-week profile, both from the last ``weeks`` weeks."""
    hist = p.sales[:, origin - 7 * weeks:origin].reshape(len(p.sales), weeks, 7)
    profile = hist.mean(1)
    level = profile.mean(1, keepdims=True)
    shape = np.where(level > 0, profile / np.maximum(level, 1e-9), 1.0)
    shrunk = 0.5 * shape + 0.5  # shrink noisy item-level weekly profiles toward flat
    return np.tile(level * shrunk, HORIZON // 7)


def exponential_smoothing(p: M5Panel, origin: int, history: int = 365) -> np.ndarray:
    """statsforecast AutoETS (weekly seasonality) per series, the classic M5 benchmark family."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    start = origin - history
    shelf = p.on_shelf()[:, start:origin]
    rows = []
    for i in range(len(p.sales)):
        y = p.sales[i, start:origin][shelf[i]]
        if len(y) < 30:
            y = np.concatenate([np.zeros(30 - len(y)), y])
        rows.append(pd.DataFrame({"unique_id": i, "ds": np.arange(len(y)), "y": y}))
    df = pd.concat(rows, ignore_index=True)
    sf = StatsForecast(models=[AutoETS(season_length=7)], freq=1, n_jobs=4)
    fc = sf.forecast(df=df, h=HORIZON)
    fc = fc.sort_values(["unique_id", "ds"])
    return np.clip(fc["AutoETS"].to_numpy().reshape(len(p.sales), HORIZON), 0, None)


def poisson_quantiles(mean: np.ndarray, taus) -> dict[float, np.ndarray]:
    return {t: stats.poisson.ppf(t, np.maximum(mean, 1e-6)) for t in taus}


# --------------------------------------------------------------------------- LightGBM

POINT = dict(
    objective="regression", learning_rate=0.06, num_leaves=255, min_data_in_leaf=100,
    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=0.1, max_bin=127,
    seed=SEED, verbose=-1, num_threads=os.cpu_count(),
)
QUANTILE = dict(
    objective="quantile", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, max_bin=127,
    seed=SEED, verbose=-1, num_threads=os.cpu_count(),
)


def _level(frame: pd.DataFrame) -> np.ndarray:
    """Per-row demand level used to make the target scale-free (recent 28-day mean + 0.5)."""
    return frame["rmean_28"].fillna(0).to_numpy() + 0.5


class GlobalLGBM:
    """One horizon-aware model across all 8k series, refit at every backtest origin.

    Two design choices fix the level bias of a plain lag-28 model:
    * horizon-aware features: each training row gets a random horizon h in 1..28 and
      sees sales up to d - h, so forecasts use the most recent data available;
    * scale-free target: the model learns demand / recent level (L2, weighted by the
      level), which transfers across fast and slow movers and across level shifts.
    """

    def __init__(self, fb: FeatureBuilder, rounds: int = 500, quantiles=(), q_rounds: int = 300, q_sample: float = 0.3):
        self.fb, self.rounds, self.quantiles, self.q_rounds, self.q_sample = fb, rounds, quantiles, q_rounds, q_sample
        self.timings: dict[str, float] = {}

    def fit_predict(self, origin: int):
        train = self.fb.build(np.arange(origin - TRAIN_DAYS, origin), random_horizon_seed=SEED)
        test = self.fb.build(np.arange(origin, origin + HORIZON), with_target=False, origin=origin)
        cols = feature_columns(train)
        lvl_tr, lvl_te = _level(train), _level(test)
        dtrain = lgb.Dataset(train[cols], train["y"].to_numpy() / lvl_tr, weight=lvl_tr,
                             categorical_feature=CATEGORICAL, free_raw_data=False)

        t = time.time()
        point = lgb.train(POINT, dtrain, num_boost_round=self.rounds)
        self.timings["point_fit_s"] = time.time() - t
        self.importance = pd.Series(point.feature_importance("gain"), index=cols).sort_values(ascending=False)
        out = {"mean": self._to_matrix(test, point.predict(test[cols]) * lvl_te)}

        # Quantile leaves are refit by a mostly serial percentile step that scales with row
        # count, so the quantile models train on a fixed 30% row sample.
        rows = np.random.default_rng(SEED).random(len(train)) < self.q_sample
        dq = lgb.Dataset(train.loc[rows, cols], train["y"].to_numpy()[rows] / lvl_tr[rows], weight=lvl_tr[rows],
                         categorical_feature=CATEGORICAL)
        for q in self.quantiles:
            t = time.time()
            m = lgb.train({**QUANTILE, "alpha": q}, dq, num_boost_round=self.q_rounds)
            self.timings[f"q{q}_fit_s"] = time.time() - t
            out[q] = self._to_matrix(test, m.predict(test[cols]) * lvl_te)
        self.n_train_rows = len(train)
        return out

    def _to_matrix(self, frame: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        """Scatter predictions back to (series, horizon); off-shelf cells stay 0."""
        n = self.fb.p.sales.shape[0]
        mat = np.zeros((n, HORIZON), dtype=np.float32)
        origin = frame["day"].min()
        mat[frame["series"].to_numpy(), frame["day"].to_numpy() - origin] = np.clip(pred, 0, None)
        return mat


def negative_binomial_quantiles(mean: np.ndarray, p: M5Panel, origin: int, taus, window: int = 112):
    """Quantiles of a negative binomial centred on the model mean.

    Dispersion is estimated per series by the method of moments on its last
    ``window`` days: Var = mu + mu^2 / k. Series that are not over-dispersed
    fall back to (near-)Poisson.
    """
    hist = p.sales[:, origin - window:origin]
    shelf = p.on_shelf()[:, origin - window:origin]
    n = np.maximum(shelf.sum(1), 1)
    m = np.where(shelf, hist, 0).sum(1) / n
    v = np.where(shelf, (hist - m[:, None]) ** 2, 0).sum(1) / np.maximum(n - 1, 1)
    excess = v - m
    k = np.where(excess > 1e-6, m ** 2 / np.maximum(excess, 1e-6), 1e4)
    k = np.clip(k, 0.05, 1e4)[:, None]
    mu = np.maximum(mean, 1e-6)
    prob = k / (k + mu)
    return {t: stats.nbinom.ppf(t, k, prob) for t in taus}

"""Leak-free features for a *direct* 28-day-ahead global model.

Every sales-derived feature for target day d uses data up to day d-28 at the
latest. A single model can therefore forecast all 28 horizons from one origin
without recursion, and nothing from inside the forecast window ever leaks in.
Prices and calendar events are known in advance (planned), so they use day d.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import HORIZON
from .data import M5Panel

WINDOWS = (7, 28, 56, 112, 364)
CATEGORICAL = ["item", "store", "state", "event_type"]


class _Cumulative:
    """Prefix sums so any trailing-window statistic is O(1) per cell."""

    def __init__(self, sales: np.ndarray, on_shelf: np.ndarray):
        x = np.where(on_shelf, sales, 0.0).astype(np.float64)
        pad = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)  # noqa: E731
        self.s = pad(x)
        self.s2 = pad(x ** 2)
        self.n = pad(on_shelf.astype(np.float64))
        self.z = pad(((x == 0) & on_shelf).astype(np.float64))

    def window(self, end: np.ndarray, w: int):
        """Stats over days (end-w, end], i.e. w days ending at ``end`` inclusive."""
        hi, lo = end + 1, np.maximum(end + 1 - w, 0)
        r = np.arange(self.s.shape[0])
        n = self.n[r, hi] - self.n[r, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (self.s[r, hi] - self.s[r, lo]) / n
            var = (self.s2[r, hi] - self.s2[r, lo]) / n - mean ** 2
            zero = (self.z[r, hi] - self.z[r, lo]) / n
        return mean, np.sqrt(np.clip(var, 0, None)), zero


def _days_since_sale(sales: np.ndarray) -> np.ndarray:
    t = np.arange(sales.shape[1])
    last = np.maximum.accumulate(np.where(sales > 0, t, -10_000), axis=1)
    return (t - last).astype(np.float32)


class FeatureBuilder:
    def __init__(self, panel: M5Panel):
        self.p = panel
        self.on_shelf = panel.on_shelf()
        self.cum = _Cumulative(panel.sales, self.on_shelf)
        self.since_sale = _days_since_sale(panel.sales)
        self.codes = {c: pd.factorize(panel.ids[f"{c}_id"])[0] for c in ("item", "store", "state")}
        cal = panel.calendar
        self.event_type = pd.factorize(cal["event_type_1"].fillna("none"))[0]
        self.snap = {s: cal[f"snap_{s}"].to_numpy() for s in ("CA", "TX", "WI")}
        price = panel.price.astype(np.float64)
        filled = np.where(np.isnan(price), 0, price)
        self.price_cum = np.cumsum(filled, axis=1)
        self.price_cnt = np.cumsum(~np.isnan(price), axis=1)

    def build(self, days: np.ndarray, with_target: bool = True, origin: int | None = None,
              random_horizon_seed: int | None = None) -> pd.DataFrame:
        """Feature frame for all on-shelf (series, day) pairs with day in ``days``.

        Each cell has a horizon h (days between the last usable observation and
        the target day). Sales features use data up to day d - h only.
        * forecasting from ``origin``: h = d - origin + 1, so day-1 forecasts use
          yesterday's sales and day-28 forecasts use the same data;
        * training with ``random_horizon_seed``: h is drawn uniformly from 1..28
          per row, so one model learns every horizon from the same rows;
        * default: h = 28 everywhere (the conservative, horizon-agnostic setup).
        """
        p, n, m = self.p, self.p.sales.shape[0], len(days)
        D = np.broadcast_to(days, (n, m))
        R = np.broadcast_to(np.arange(n)[:, None], (n, m))
        if origin is not None:
            H = D - origin + 1
        elif random_horizon_seed is not None:
            H = np.random.default_rng(random_horizon_seed).integers(1, HORIZON + 1, size=(n, m))
        else:
            H = np.full((n, m), HORIZON)
        ref = D - H  # last day whose sales may be used
        feats: dict[str, np.ndarray] = {"horizon": H.astype(np.float32)}

        # Same-weekday lags: the k-th most recent observed day sharing the target's weekday.
        first_sw = 7 * np.ceil(H / 7).astype(int)
        for k in range(4):
            back = first_sw + 7 * k
            feats[f"same_wday_lag_{k + 1}"] = np.where(self.on_shelf[R, D - back], p.sales[R, D - back], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN before an item's launch
            feats["same_wday_mean_4"] = np.nanmean(np.stack([feats[f"same_wday_lag_{k + 1}"] for k in range(4)]), axis=0)
        feats["last_value"] = np.where(self.on_shelf[R, ref], p.sales[R, ref], np.nan)
        for lag in (364,):
            lagged = p.sales[R, D - lag].astype(np.float32)
            feats[f"lag_{lag}"] = np.where(self.on_shelf[R, D - lag], lagged, np.nan)
        for w in WINDOWS:
            mean, std, zero = (np.stack(a, axis=1) for a in zip(*[self.cum.window(ref[:, j], w) for j in range(m)]))
            feats[f"rmean_{w}"] = mean
            if w in (28, 112):
                feats[f"rstd_{w}"] = std
                feats[f"zero_share_{w}"] = zero
        feats["days_since_sale"] = self.since_sale[R, ref]
        feats["trend_28_112"] = feats["rmean_28"] / (feats["rmean_112"] + 0.1)

        price = p.price[R, D]
        hist_mean = self.price_cum[R, ref] / np.maximum(self.price_cnt[R, ref], 1)
        feats["price"] = price
        feats["price_rel_hist"] = price / np.where(hist_mean > 0, hist_mean, np.nan)
        feats["price_change_7"] = price / p.price[R, D - 7] - 1

        cal = p.calendar
        dates = cal["date"].to_numpy()[days]
        dow = pd.DatetimeIndex(dates)
        feats["wday"] = np.broadcast_to(dow.dayofweek.to_numpy(), (n, m))
        feats["mday"] = np.broadcast_to(dow.day.to_numpy(), (n, m))
        feats["month"] = np.broadcast_to(dow.month.to_numpy(), (n, m))
        feats["event_type"] = np.broadcast_to(self.event_type[days], (n, m))
        feats["event_tomorrow"] = np.broadcast_to(self.event_type[np.minimum(days + 1, len(cal) - 1)] > 0, (n, m))
        state = p.ids["state_id"].to_numpy()
        feats["snap"] = np.stack([self.snap[s][days] for s in state])
        for c, codes in self.codes.items():
            feats[c] = np.broadcast_to(codes[:, None], (n, m))

        keep = self.on_shelf[R, D].ravel()
        frame = pd.DataFrame({k: np.asarray(v, dtype=np.float32).ravel()[keep] for k, v in feats.items()})
        for c in CATEGORICAL:
            frame[c] = frame[c].astype(np.int32)
        frame["series"] = R.ravel()[keep].astype(np.int32)
        frame["day"] = D.ravel()[keep].astype(np.int32)
        if with_target:
            frame["y"] = p.sales[R, D].ravel()[keep].astype(np.float32)
        return frame


FEATURES_EXCLUDE = ("series", "day", "y")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in FEATURES_EXCLUDE]

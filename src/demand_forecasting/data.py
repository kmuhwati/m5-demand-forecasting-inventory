"""Load M5 into dense (series x day) matrices.

Working on wide numpy arrays instead of a long dataframe makes lag and rolling
features vectorised and keeps memory small (8,230 series x 1,941 days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEPT, PROCESSED, RAW


@dataclass
class M5Panel:
    ids: pd.DataFrame          # one row per series: id, item_id, store_id, state_id, ...
    sales: np.ndarray          # (n_series, n_days) float32 unit sales
    price: np.ndarray          # (n_series, n_days) float32 sell price, NaN when not on shelf
    calendar: pd.DataFrame     # one row per day, index 0 == d_1

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    def on_shelf(self) -> np.ndarray:
        """True once an item has a listed price at that store (before that it was not sold)."""
        return ~np.isnan(self.price)


def _price_matrix(ids: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.item_id.str.startswith(DEPT)]
    rows = ids.reset_index()[["index", "item_id", "store_id"]].rename(columns={"index": "row"})
    prices = prices.merge(rows, on=["item_id", "store_id"], how="inner")

    # Each Walmart week (wm_yr_wk) covers consecutive calendar days.
    days = calendar.reset_index()[["index", "wm_yr_wk"]].rename(columns={"index": "day"})
    long = prices.merge(days, on="wm_yr_wk", how="inner")
    mat = np.full((len(ids), len(calendar)), np.nan, dtype=np.float32)
    mat[long["row"].to_numpy(), long["day"].to_numpy()] = long["sell_price"].to_numpy(np.float32)
    return mat


def load(cache: bool = True) -> M5Panel:
    cache_file = PROCESSED / f"{DEPT.lower()}_panel.npz"
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    if cache and cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        ids = pd.DataFrame(z["ids"], columns=z["id_cols"])
        return M5Panel(ids, z["sales"], z["price"], calendar)

    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[sales.dept_id == DEPT].reset_index(drop=True)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ids = sales[id_cols].copy()
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    calendar = calendar.iloc[: len(day_cols)].reset_index(drop=True)
    mat = sales[day_cols].to_numpy(np.float32)
    price = _price_matrix(ids, calendar)

    if cache:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, ids=ids.to_numpy(), id_cols=np.array(id_cols), sales=mat, price=price)
    return M5Panel(ids, mat, price, calendar)

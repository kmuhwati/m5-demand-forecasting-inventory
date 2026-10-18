"""Replay a daily order-up-to policy against realised demand.

Each morning the store tops inventory up to a target level S_t (zero lead
time); demand that exceeds stock is lost; leftover stock carries over and
incurs a holding cost. The newsvendor result says the cost-minimising S_t is
the critical-ratio quantile of tomorrow's demand, which is why probabilistic
forecasts matter more for this decision than point accuracy does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(order_up_to: np.ndarray, demand: np.ndarray, price: np.ndarray,
             margin: float, holding_rate: float) -> dict:
    """Vectorised over series; loops over the days of the horizon."""
    n, h = demand.shape
    stock = np.zeros(n)
    sold = np.zeros((n, h))
    held = np.zeros((n, h))
    ordered = np.zeros((n, h))
    for d in range(h):
        target = np.ceil(np.maximum(order_up_to[:, d], 0))
        order = np.maximum(target - stock, 0)
        stock = stock + order
        sold[:, d] = np.minimum(stock, demand[:, d])
        stock = stock - sold[:, d]
        held[:, d] = stock
        ordered[:, d] = order
    revenue_margin = margin * np.nansum(sold * price)
    holding_cost = holding_rate * np.nansum(held * price)
    lost = demand - sold
    return {
        "profit": float(revenue_margin - holding_cost),
        "gross_margin": float(revenue_margin),
        "holding_cost": float(holding_cost),
        "fill_rate": float(sold.sum() / max(demand.sum(), 1)),
        "lost_units": float(lost.sum()),
        "avg_units_on_hand": float(held.mean()),
        "units_ordered": float(ordered.sum()),
    }


def summarise(results: dict[str, dict], perfect_key: str = "Perfect foresight") -> pd.DataFrame:
    df = pd.DataFrame(results).T
    if perfect_key in df.index:
        df["share_of_perfect_profit"] = df["profit"] / df.loc[perfect_key, "profit"]
    return df

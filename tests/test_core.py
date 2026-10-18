"""Tests on tiny synthetic panels where the right answer is known."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecasting import inventory, metrics, models  # noqa: E402
from demand_forecasting.config import HORIZON  # noqa: E402
from demand_forecasting.data import M5Panel  # noqa: E402
from demand_forecasting.features import FeatureBuilder  # noqa: E402


def tiny_panel(n_series=3, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    sales = rng.poisson(3, size=(n_series, n_days)).astype(np.float32)
    price = np.full((n_series, n_days), 2.0, dtype=np.float32)
    price[0, :50] = np.nan  # item 0 launched on day 50
    ids = pd.DataFrame({
        "id": [f"FOODS_3_{i:03d}_CA_1_evaluation" for i in range(n_series)],
        "item_id": [f"FOODS_3_{i:03d}" for i in range(n_series)],
        "dept_id": "FOODS_3", "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
    })
    dates = pd.date_range("2015-01-01", periods=n_days)
    cal = pd.DataFrame({"date": dates, "event_type_1": np.nan, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0})
    return M5Panel(ids, sales, price, cal)


def test_features_never_use_data_inside_horizon():
    p = tiny_panel()
    fb = FeatureBuilder(p)
    days = np.arange(450, 450 + HORIZON)
    before = fb.build(days)
    # Overwrite everything from the forecast origin onward; features must not change.
    p.sales[:, 450:] = 999
    after = FeatureBuilder(p).build(days)
    cols = [c for c in before.columns if c not in ("y",)]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_lag_feature_alignment():
    p = tiny_panel()
    f = FeatureBuilder(p).build(np.array([400]))          # default horizon 28
    row = f[f.series == 1].iloc[0]
    assert row.same_wday_lag_1 == p.sales[1, 372]
    assert row.rmean_7 == pytest.approx(p.sales[1, 366:373].mean(), rel=1e-5)


def test_horizon_aware_features_use_latest_data_but_nothing_after_origin():
    p = tiny_panel()
    fb = FeatureBuilder(p)
    f = fb.build(np.arange(450, 450 + HORIZON), with_target=False, origin=450)
    day1 = f[(f.series == 1) & (f.day == 450)].iloc[0]
    day10 = f[(f.series == 1) & (f.day == 459)].iloc[0]
    assert day1.horizon == 1 and day10.horizon == 10
    assert day1.last_value == p.sales[1, 449] and day10.last_value == p.sales[1, 449]
    assert day10.same_wday_lag_1 == p.sales[1, 459 - 14]  # first same-weekday day at or before the origin
    p.sales[:, 450:] = 999
    after = FeatureBuilder(p).build(np.arange(450, 450 + HORIZON), with_target=False, origin=450)
    pd.testing.assert_frame_equal(f, after)


def test_off_shelf_rows_are_dropped():
    p = tiny_panel()
    f = FeatureBuilder(p).build(np.arange(30, 60))
    assert f[(f.series == 0)].day.min() == 50


def test_seasonal_naive_repeats_last_week():
    p = tiny_panel()
    fc = models.seasonal_naive(p, 400)
    assert np.array_equal(fc[:, :7], p.sales[:, 393:400])
    assert np.array_equal(fc[:, 7:14], p.sales[:, 393:400])


def test_negative_binomial_quantiles_are_monotone_and_overdispersed():
    p = tiny_panel()
    mean = np.full((3, HORIZON), 3.0)
    q = models.negative_binomial_quantiles(mean, p, 400, [0.5, 0.9, 0.98])
    assert (q[0.5] <= q[0.9]).all() and (q[0.9] <= q[0.98]).all()


def test_rmsse_zero_for_perfect_forecast():
    y = np.array([[1.0, 2.0, 3.0]])
    assert metrics.rmsse(y, y, np.array([1.0]))[0] == 0


def test_pinball_penalises_under_forecast_more_at_high_tau():
    y = np.array([[10.0]])
    s = np.array([1.0])
    assert metrics.scaled_pinball(y, np.array([[5.0]]), 0.9, s)[0] > metrics.scaled_pinball(y, np.array([[15.0]]), 0.9, s)[0]


def test_inventory_perfect_foresight_has_full_fill_and_no_holding():
    demand = np.array([[3.0, 0.0, 5.0]])
    r = inventory.simulate(demand, demand, np.ones_like(demand), margin=0.3, holding_rate=0.1)
    assert r["fill_rate"] == 1.0 and r["holding_cost"] == 0.0
    assert r["profit"] == pytest.approx(0.3 * 8)


def test_inventory_carries_over_stock():
    demand = np.array([[1.0, 1.0]])
    r = inventory.simulate(np.array([[3.0, 0.0]]), demand, np.ones_like(demand), margin=0.3, holding_rate=0.1)
    assert r["units_ordered"] == 3.0 and r["fill_rate"] == 1.0


def test_intermittency_classes():
    smooth = np.full((1, 100), 5.0)
    lumpy = np.zeros((1, 100))
    lumpy[0, ::10] = [1, 20, 1, 30, 2, 25, 1, 40, 1, 10]
    shelf = np.ones((1, 100), dtype=bool)
    assert metrics.intermittency_class(smooth, shelf)[0] == "smooth"
    assert metrics.intermittency_class(lumpy, shelf)[0] == "lumpy"

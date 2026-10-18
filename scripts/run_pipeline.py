"""Rolling-origin backtest: point + probabilistic forecasts -> inventory replay.

Run from the project root:  python scripts/run_pipeline.py
Outputs: reports/figures, reports/tables, reports/metrics.json, reports/RESULTS.md
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from demand_forecasting import data, inventory, metrics, models, plotstyle  # noqa: E402
from demand_forecasting.config import ECON, FIGURES, HORIZON, ORIGINS, PROCESSED, QUANTILES, REPORTS, TABLES  # noqa: E402
from demand_forecasting.features import FeatureBuilder  # noqa: E402

TAU_GRID = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    plotstyle.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    log("loading panel")
    p = data.load()
    fb = FeatureBuilder(p)
    shelf = p.on_shelf()
    groups = {"state": pd.factorize(p.ids.state_id)[0], "store": pd.factorize(p.ids.store_id)[0]}
    cr = ECON.critical_ratio
    out: dict = {"n_series": int(len(p.sales)), "origins": list(ORIGINS), "critical_ratio": cr}

    point_rows, prob_rows, inv_rows, curve_rows, class_rows, timing_rows = [], [], [], [], [], []
    keep_for_plots = {}
    importances = []

    for fold, origin in enumerate(ORIGINS, start=1):
        window = f"d_{origin + 1}..d_{origin + HORIZON}"
        log(f"fold {fold}: forecasting {window}")
        y = p.sales[:, origin:origin + HORIZON]
        hist, hist_shelf = p.sales[:, :origin], shelf[:, :origin]
        price_te = np.nan_to_num(p.price[:, origin:origin + HORIZON])
        abs_scale = metrics.naive_abs_scale(hist, hist_shelf)

        t = time.time()
        preds = {
            "Seasonal naive (last week)": models.seasonal_naive(p, origin),
            "Moving average (28d)": models.moving_average(p, origin),
            "Weekday-profile moving average": models.weekday_profile_ma(p, origin),
        }
        # Expensive model fits are cached per origin so a later failure never costs a refit.
        ets_cache, gbm_cache = PROCESSED / f"ets_{origin}.pkl", PROCESSED / f"lgbm_{origin}.pkl"
        if ets_cache.exists():
            log("  AutoETS (cached)")
            ets_c = pd.read_pickle(ets_cache)
        else:
            log("  AutoETS")
            t_ets = time.time()
            ets_c = {"ets": models.exponential_smoothing(p, origin), "seconds": time.time() - t_ets}
            pd.to_pickle(ets_c, ets_cache)
        if gbm_cache.exists():
            log("  LightGBM (cached)")
            gbm_c = pd.read_pickle(gbm_cache)
        else:
            log("  LightGBM (horizon-aware point + quantiles)")
            gbm = models.GlobalLGBM(fb, quantiles=QUANTILES)
            gbm_c = {"lg": gbm.fit_predict(origin), "timings": gbm.timings,
                     "importance": gbm.importance, "n_train_rows": gbm.n_train_rows}
            pd.to_pickle(gbm_c, gbm_cache)
        c = {"ets": ets_c["ets"], "lg": gbm_c["lg"], "timings": {"AutoETS": ets_c["seconds"], **gbm_c["timings"]},
             "importance": gbm_c["importance"], "n_train_rows": gbm_c["n_train_rows"]}
        preds["AutoETS (statsforecast)"] = c["ets"]
        lg = c["lg"]
        preds["LightGBM global"] = lg["mean"]
        timing_rows += [{"fold": fold, "model": k, "seconds": v} for k, v in c["timings"].items()]
        importances.append(c["importance"].rename(f"fold{fold}"))
        out.setdefault("train_rows", []).append(c["n_train_rows"])

        # ---------------- point accuracy
        for name, yhat in preds.items():
            w = metrics.wrmsse_levels(y, yhat, hist, hist_shelf, p.price[:, :origin], groups)
            err = yhat - y
            point_rows.append({"fold": fold, "model": name, **{f"wrmsse_{k}": v for k, v in w.items()},
                               "mae": float(np.abs(err).mean()), "bias_pct": float(100 * err.sum() / y.sum())})
            log(f"    {name:34s} WRMSSE(avg)={w['average']:.4f}")

        # ---------------- probabilistic accuracy
        prob = {
            "Poisson around 28d moving average": models.poisson_quantiles(preds["Moving average (28d)"], TAU_GRID),
            "Negative binomial around LightGBM mean": models.negative_binomial_quantiles(lg["mean"], p, origin, TAU_GRID),
            "LightGBM quantile regression": {q: lg[q] for q in QUANTILES},
        }
        for name, qs in prob.items():
            for tau, q in qs.items():
                spl = metrics.scaled_pinball(y, q, tau, abs_scale)
                prob_rows.append({"fold": fold, "method": name, "tau": tau,
                                  "scaled_pinball": float(np.nanmean(spl)), "coverage": metrics.coverage(y, q)})

        # ---------------- inventory replay
        policies = {
            "Perfect foresight": y,
            "Seasonal naive point": preds["Seasonal naive (last week)"],
            "Moving-average point": preds["Moving average (28d)"],
            "AutoETS point": preds["AutoETS (statsforecast)"],
            "LightGBM point (mean)": lg["mean"],
            f"Poisson MA, q={cr:.2f}": models.poisson_quantiles(preds["Moving average (28d)"], [cr])[cr],
            f"LightGBM + NB, q={cr:.2f}": models.negative_binomial_quantiles(lg["mean"], p, origin, [cr])[cr],
            f"LightGBM quantile, q={cr:.2f}": lg[cr],
        }
        for name, S in policies.items():
            r = inventory.simulate(S, y, price_te, ECON.gross_margin, ECON.holding_rate)
            inv_rows.append({"fold": fold, "policy": name, **r})
        for tau, S in prob["Negative binomial around LightGBM mean"].items():
            r = inventory.simulate(S, y, price_te, ECON.gross_margin, ECON.holding_rate)
            curve_rows.append({"fold": fold, "method": "LightGBM + NB", "tau": tau, **r})
        for tau in QUANTILES:
            r = inventory.simulate(lg[tau], y, price_te, ECON.gross_margin, ECON.holding_rate)
            curve_rows.append({"fold": fold, "method": "LightGBM quantile", "tau": tau, **r})

        # ---------------- error by demand class
        classes = metrics.intermittency_class(hist[:, -365:], hist_shelf[:, -365:])
        sq_scale = metrics.naive_scale(hist, hist_shelf)
        for name in ("Moving average (28d)", "AutoETS (statsforecast)", "LightGBM global"):
            r = metrics.rmsse(y, preds[name], sq_scale)
            for c in np.unique(classes):
                class_rows.append({"fold": fold, "model": name, "demand_class": c,
                                   "n_series": int((classes == c).sum()), "median_rmsse": float(np.nanmedian(r[classes == c]))})

        if fold == len(ORIGINS):
            keep_for_plots = dict(origin=origin, y=y, preds=preds, nb=prob["Negative binomial around LightGBM mean"],
                                  lq={q: lg[q] for q in QUANTILES})
        log(f"  fold done in {time.time() - t:.0f}s")

    # =============================================================== aggregate
    point = pd.DataFrame(point_rows)
    prob = pd.DataFrame(prob_rows)
    inv = pd.DataFrame(inv_rows)
    curve = pd.DataFrame(curve_rows)
    classes = pd.DataFrame(class_rows)
    for name, frame in [("point_accuracy_by_fold", point), ("probabilistic_by_fold", prob), ("inventory_by_fold", inv),
                        ("service_level_curve", curve), ("rmsse_by_demand_class", classes),
                        ("timings", pd.DataFrame(timing_rows))]:
        frame.to_csv(TABLES / f"{name}.csv", index=False)

    point_summary = point.drop(columns="fold").groupby("model").mean().sort_values("wrmsse_average")
    best_baseline = point_summary.drop("LightGBM global").iloc[0]
    lgb_row = point_summary.loc["LightGBM global"]
    prob_summary = prob.groupby(["method", "tau"])[["scaled_pinball", "coverage"]].mean().reset_index()
    inv_summary = inv.drop(columns="fold").groupby("policy").sum()
    inv_summary["share_of_perfect_profit"] = inv_summary["profit"] / inv_summary.loc["Perfect foresight", "profit"]
    inv_summary["fill_rate"] = inv.groupby("policy")["fill_rate"].mean()
    inv_summary = inv_summary.sort_values("profit", ascending=False)
    curve_summary = curve.groupby(["method", "tau"])[["profit", "fill_rate", "avg_units_on_hand"]].agg(
        {"profit": "sum", "fill_rate": "mean", "avg_units_on_hand": "mean"}).reset_index()
    imp = pd.concat(importances, axis=1).mean(1).sort_values(ascending=False)
    imp = imp / imp.sum()

    point_summary.to_csv(TABLES / "point_accuracy_summary.csv")
    prob_summary.to_csv(TABLES / "probabilistic_summary.csv", index=False)
    inv_summary.to_csv(TABLES / "inventory_summary.csv")
    imp.to_csv(TABLES / "feature_importance.csv")

    best_point = inv_summary.loc[["Seasonal naive point", "Moving-average point", "AutoETS point", "LightGBM point (mean)"], "profit"].max()
    best_policy = inv_summary.drop("Perfect foresight").iloc[0]
    out.update({
        "wrmsse": point_summary["wrmsse_average"].round(4).to_dict(),
        "lightgbm_vs_best_baseline_wrmsse_pct": float(100 * (1 - lgb_row.wrmsse_average / best_baseline.wrmsse_average)),
        "best_baseline": best_baseline.name,
        "inventory": inv_summary[["profit", "share_of_perfect_profit", "fill_rate", "lost_units", "avg_units_on_hand"]].round(4).to_dict(orient="index"),
        "best_policy": best_policy.name,
        "best_policy_profit_uplift_vs_best_point_pct": float(100 * (best_policy.profit / best_point - 1)),
        "coverage_at_critical_ratio": prob_summary[np.isclose(prob_summary.tau, cr)].set_index("method")["coverage"].round(4).to_dict(),
    })
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    # =============================================================== figures
    order = point_summary.index[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [plotstyle.SERIES[0] if "LightGBM" in m else plotstyle.NEUTRAL for m in order]
    ax.barh(order, point_summary.loc[order, "wrmsse_average"], color=colors, height=0.6)
    for i, m in enumerate(order):
        vals = point[point.model == m].wrmsse_average
        ax.scatter(vals, [i] * len(vals), color=plotstyle.INK_2, s=12, zorder=3)
        ax.annotate(f"{point_summary.loc[m, 'wrmsse_average']:.3f}", (point_summary.loc[m, "wrmsse_average"], i),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("WRMSSE, average of total/state/store/item levels (lower is better); dots = folds")
    ax.set_title(f"Global LightGBM beats the best baseline by {out['lightgbm_vs_best_baseline_wrmsse_pct']:.0f}%")
    plotstyle.save(fig, FIGURES / "01_point_accuracy.png")

    k = keep_for_plots
    totals = {m: k["preds"][m].sum(0) for m in ("AutoETS (statsforecast)", "LightGBM global")}
    hist_total = p.sales[:, k["origin"] - 56:k["origin"]].sum(0)
    dates = p.calendar.date.iloc[k["origin"] - 56:k["origin"] + HORIZON].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dates[:56], hist_total, color=plotstyle.INK_2, lw=1.5, label="History")
    ax.plot(dates[56:], k["y"].sum(0), color=plotstyle.INK, lw=1.5, label="Actual")
    ax.plot(dates[56:], totals["LightGBM global"], color=plotstyle.SERIES[0], label="LightGBM")
    ax.plot(dates[56:], totals["AutoETS (statsforecast)"], color=plotstyle.SERIES[1], ls="--", label="AutoETS")
    ax.set_ylabel("Units / day, FOODS_3, 10 stores")
    ax.set_title("Final fold (M5 public-leaderboard window, 25 Apr - 22 May 2016): department total")
    ax.legend(loc="upper left", ncol=4)
    plotstyle.save(fig, FIGURES / "02_department_total_forecast.png")

    yy = k["y"]
    top = np.argsort(-yy.sum(1))[[0, 50, 400]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    h = np.arange(1, HORIZON + 1)
    for ax, i in zip(axes, top):
        ax.fill_between(h, k["nb"][0.5][i], k["nb"][0.95][i], color=plotstyle.SERIES[0], alpha=0.12, lw=0, label="median to q95 (NB)")
        ax.fill_between(h, k["nb"][0.75][i], k["nb"][0.9][i], color=plotstyle.SERIES[0], alpha=0.18, lw=0, label="q75 to q90 (NB)")
        ax.plot(h, k["preds"]["LightGBM global"][i], color=plotstyle.SERIES[0], label="Mean forecast")
        ax.plot(h, yy[i], color=plotstyle.INK, marker="o", ms=3, lw=1, label="Actual")
        ax.set_title(p.ids.id.iloc[i].replace("_evaluation", ""), fontsize=10)
        ax.set_xlabel("Days ahead")
    axes[0].set_ylabel("Units sold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Probabilistic forecasts for a fast, a medium and a slow mover", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plotstyle.save(fig, FIGURES / "03_fan_charts.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for j, (name, g) in enumerate(prob_summary.groupby("method")):
        ax.plot(g.tau, g.coverage, marker="o", color=plotstyle.SERIES[j], label=name)
    ax.plot([0.45, 1], [0.45, 1], color=plotstyle.AXIS, ls=":", lw=1, label="Perfect calibration")
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Share of actuals at or below the quantile")
    ax.set_title("Calibration of predictive quantiles")
    ax.legend(fontsize=8, loc="upper left")
    plotstyle.save(fig, FIGURES / "04_calibration.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    pol = inv_summary.drop("Perfect foresight")
    names = pol.index[::-1]
    colors = [plotstyle.SERIES[2] if "q=" in n and "LightGBM" in n else (plotstyle.SERIES[0] if "LightGBM" in n else plotstyle.NEUTRAL) for n in names]
    ax.barh(names, 100 * pol.loc[names, "share_of_perfect_profit"], color=colors, height=0.6)
    for i, n in enumerate(names):
        ax.annotate(f"{100 * pol.loc[n, 'share_of_perfect_profit']:.1f}%  (fill {100 * pol.loc[n, 'fill_rate']:.0f}%)",
                    (100 * pol.loc[n, "share_of_perfect_profit"], i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)
    ax.set_xlabel("Profit as % of perfect-foresight profit (3 folds)")
    best_name = pol.index[0]
    ax.set_title("Share of perfect-foresight profit by ordering policy")
    ax.set_xlim(0, 115)
    ax = axes[1]
    for j, (name, g) in enumerate(curve_summary.groupby("method")):
        ax.plot(g.tau, g.profit / 1e3, marker="o", color=plotstyle.SERIES[j], label=name)
    ax.axvline(cr, color=plotstyle.AXIS, ls=":", lw=1)
    ax.annotate(f"critical ratio {cr:.2f}", (cr, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points", fontsize=8, color=plotstyle.INK_2)
    ax.set_xlabel("Quantile used as order-up-to level")
    ax.set_ylabel("Profit over 3 folds ($ thousands)")
    peak = curve_summary.loc[curve_summary.profit.idxmax()]
    ax.set_title(f"Profit peaks at q={peak.tau:.2f} (critical ratio {cr:.2f})")
    ax.legend()
    plotstyle.save(fig, FIGURES / "05_inventory.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    t15 = imp.head(15)[::-1]
    ax.barh(t15.index, 100 * t15.values, color=plotstyle.SERIES[0], height=0.6)
    ax.set_xlabel("Share of total split gain (%), averaged over folds")
    ax.set_title("What the global model relies on")
    plotstyle.save(fig, FIGURES / "06_feature_importance.png")

    cs = classes.groupby(["model", "demand_class"])["median_rmsse"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(cs))
    for j, m in enumerate(cs.columns):
        ax.bar(x + (j - 1) * 0.27, cs[m], width=0.25, color=[plotstyle.NEUTRAL, plotstyle.SERIES[1], plotstyle.SERIES[0]][j], label=m)
    counts = classes.groupby("demand_class")["n_series"].mean()
    ax.set_xticks(x, [f"{c}\n(n={counts[c]:.0f})" for c in cs.index])
    ax.set_ylabel("Median RMSSE")
    ax.set_title("Error by Syntetos-Boylan demand class")
    ax.legend(fontsize=8)
    plotstyle.save(fig, FIGURES / "07_error_by_demand_class.png")

    # =============================================================== results markdown
    md = lambda f, fmt=".4f": f.to_markdown(floatfmt=fmt)  # noqa: E731
    lines = [
        "# Results (auto-generated by scripts/run_pipeline.py)\n",
        f"Scope: {out['n_series']:,} item-store series (FOODS_3, 10 stores); 3 rolling origins x 28 days; "
        f"critical ratio {cr:.2f} (margin {ECON.gross_margin:.0%}, holding {ECON.holding_rate:.0%} of price per unit-day).\n",
        "## Point accuracy (mean over folds)\n", md(point_summary),
        "\n## Probabilistic accuracy (scaled pinball loss, empirical coverage)\n",
        prob_summary.to_markdown(index=False, floatfmt=".4f"),
        "\n## Inventory replay (summed over folds)\n", md(inv_summary, ".3f"),
        "\n## Service-level curve\n", curve_summary.to_markdown(index=False, floatfmt=".3f"),
        "\n## Median RMSSE by demand class\n", md(cs),
        "\n## Top features (share of gain)\n", imp.head(15).to_frame("share").to_markdown(floatfmt=".3f"),
        f"\n```json\n{json.dumps({k: v for k, v in out.items() if k not in ('inventory',)}, indent=2, default=float)}\n```\n",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()

"""Forecast accuracy metrics on (series x horizon) matrices.

RMSSE / WRMSSE follow the M5 competition definitions (Makridakis et al., 2022):
errors are scaled by each series' in-sample one-step naive error and weighted
by recent dollar sales, so a model is rewarded for getting the items that
matter to revenue right.
"""
from __future__ import annotations

import numpy as np


def naive_scale(history: np.ndarray, on_shelf: np.ndarray) -> np.ndarray:
    """Mean squared one-step difference from each series' first on-shelf day."""
    d = np.diff(history, axis=1) ** 2
    valid = on_shelf[:, 1:] & on_shelf[:, :-1]
    scale = np.where(valid, d, 0).sum(1) / np.maximum(valid.sum(1), 1)
    return np.where(scale > 0, scale, np.nan)


def naive_abs_scale(history: np.ndarray, on_shelf: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(history, axis=1))
    valid = on_shelf[:, 1:] & on_shelf[:, :-1]
    scale = np.where(valid, d, 0).sum(1) / np.maximum(valid.sum(1), 1)
    return np.where(scale > 0, scale, np.nan)


def dollar_weights(history: np.ndarray, price: np.ndarray, last: int = 28) -> np.ndarray:
    dollars = np.nansum(history[:, -last:] * price[:, -last:], axis=1)
    return dollars / dollars.sum()


def rmsse(y: np.ndarray, yhat: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y - yhat) ** 2, axis=1) / scale)


def wrmsse_levels(y, yhat, history, on_shelf, price, groups: dict[str, np.ndarray]) -> dict[str, float]:
    """WRMSSE at several aggregation levels plus their average.

    ``groups`` maps level name -> integer group id per series (e.g. store).
    Level 'item_store' (the bottom) is always included.
    """
    dollars_hist = np.nan_to_num(history * price)
    out = {}
    levels = {"total": np.zeros(len(y), dtype=int), **groups, "item_store": np.arange(len(y))}
    for name, g in levels.items():
        k = g.max() + 1
        agg = lambda a: np.stack([a[g == i].sum(0) for i in range(k)])  # noqa: E731
        Y, YH, H = agg(y), agg(yhat), agg(history)
        shelf = np.ones_like(H, dtype=bool) if name != "item_store" else on_shelf
        s = naive_scale(H, shelf)
        w = agg(dollars_hist[:, -28:]).sum(1)
        w = w / w.sum()
        r = rmsse(Y, YH, s)
        ok = ~np.isnan(r)
        out[name] = float(np.sum(w[ok] * r[ok]) / w[ok].sum())
    out["average"] = float(np.mean(list(out.values())))
    return out


def scaled_pinball(y: np.ndarray, q: np.ndarray, tau: float, abs_scale: np.ndarray) -> np.ndarray:
    """Per-series pinball loss scaled by in-sample naive MAE (M5 Uncertainty style)."""
    diff = y - q
    loss = np.maximum(tau * diff, (tau - 1) * diff).mean(axis=1)
    return loss / abs_scale


def coverage(y: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(y <= q))


def intermittency_class(history: np.ndarray, on_shelf: np.ndarray) -> np.ndarray:
    """Syntetos-Boylan demand classes from average inter-demand interval and CV^2 of sizes."""
    labels = np.empty(len(history), dtype=object)
    for i, (h, s) in enumerate(zip(history, on_shelf)):
        h = h[s]
        nz = h[h > 0]
        if len(nz) < 2:
            labels[i] = "lumpy"
            continue
        adi = len(h) / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2
        labels[i] = ("smooth" if cv2 < 0.49 else "erratic") if adi < 1.32 else ("intermittent" if cv2 < 0.49 else "lumpy")
    return labels

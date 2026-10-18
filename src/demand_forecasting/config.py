"""Paths, scope, backtest design and inventory economics."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEED = 7

# Scope: the largest M5 department (fresh & packaged food) across all 10 stores.
DEPT = "FOODS_3"

HORIZON = 28                       # days forecast from each origin (M5 standard)
# Rolling-origin backtest: each fold forecasts the 28 days after its origin.
# The last fold is the M5 public-leaderboard (validation) window, d_1914..d_1941
# (25 Apr - 22 May 2016); the final M5 window's actuals are not in the released file.
ORIGINS = (1857, 1885, 1913)
TRAIN_DAYS = 730                   # training window length before each origin
FIRST_FEATURE_DAY = 1913 - TRAIN_DAYS - 3 * HORIZON - 140  # warm-up for long rolling windows

QUANTILES = (0.5, 0.75, 0.9, 0.95)


@dataclass(frozen=True)
class InventoryEconomics:
    """Daily order-up-to policy with zero lead time and lost sales.

    gross_margin  - lost profit per unit of unmet demand, as a share of price
    holding_rate  - cost per unit left on the shelf overnight (capital, space,
                    spoilage of perishables), as a share of price
    The newsvendor critical ratio margin / (margin + holding) is the service
    level a cost-minimising order-up-to level should target.
    """

    gross_margin: float = 0.30
    holding_rate: float = 0.10

    @property
    def critical_ratio(self) -> float:
        return round(self.gross_margin / (self.gross_margin + self.holding_rate), 4)


ECON = InventoryEconomics()

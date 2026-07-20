"""
Backtest skill: SMA crossover vs buy-and-hold, computed honestly.

"Honestly" is load-bearing: positions are lagged one bar (no lookahead),
transaction costs are charged on every position change, and the caveats ride
inside the tool result so the model cannot summarize the numbers without
seeing them. A backtest that flatters itself is worse than no backtest.
"""

from __future__ import annotations

import logging
from typing import Any

from ._market_data import MarketDataError, as_of, get_history
from ._quant import (
    _round,
    backtest_equity,
    sma_cross_positions,
    summarize_prices,
)

logger = logging.getLogger(__name__)

CAVEATS = [
    "In-sample result: the parameters were chosen knowing this history.",
    "One instrument, one path -- no statistical significance is claimed.",
    "Signals trade at the next day's close; intraday fills would differ.",
    "Costs are a flat per-trade estimate; slippage and taxes are not modeled.",
    "Dividends are included via adjusted closes; borrowing costs are not.",
    "Past performance does not indicate future results.",
]


def backtest_sma_cross(
    ticker: str,
    fast: int = 50,
    slow: int = 200,
    years: float = 5.0,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Backtest a long-or-flat SMA crossover strategy against buy-and-hold.

    The strategy is long when the fast simple moving average is above the slow
    one, flat otherwise. Signals are lagged one day to avoid lookahead bias,
    and each position change pays a transaction cost. Results are historical
    and in-sample -- report them with the caveats included in the output, and
    never as evidence the strategy will work in the future.

    Args:
        ticker: One ticker symbol, e.g. 'SPY'.
        fast: Fast SMA window in trading days, 2-100. Defaults to 50.
        slow: Slow SMA window in trading days, 10-300, must exceed fast.
            Defaults to 200.
        years: Backtest window in years, 1 to 10. Defaults to 5.
        cost_bps: One-way transaction cost in basis points charged on each
            position change. Defaults to 10 (0.10%).

    Returns:
        A dict with 'strategy' and 'buy_and_hold' stat blocks, trade count,
        time-in-market, the caveats list, or an 'error'.
    """
    try:
        fast = int(fast)
        slow = int(slow)
        years = float(min(max(years, 1.0), 10.0))
        cost_bps = float(min(max(cost_bps, 0.0), 200.0))

        if not 2 <= fast <= 100:
            return {"status": "error", "error": "fast must be between 2 and 100."}
        if not 10 <= slow <= 300:
            return {"status": "error", "error": "slow must be between 10 and 300."}
        if fast >= slow:
            return {
                "status": "error",
                "error": f"fast ({fast}) must be smaller than slow ({slow}).",
            }

        # Fetch extra history so the slow SMA is fully formed at the window
        # start, rather than silently spending the first year flat.
        prices = get_history(ticker, years=years + slow / 252 + 0.1)
    except MarketDataError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - tool results must not raise
        logger.exception("backtest_sma_cross failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    if len(prices) < slow + 30:
        return {
            "status": "error",
            "error": (
                f"Only {len(prices)} trading days for '{ticker.upper()}'; "
                f"need at least {slow + 30} for a {slow}-day SMA backtest."
            ),
        }

    positions = sma_cross_positions(prices, fast, slow)

    # Evaluate both legs on the same post-warmup window.
    start = prices.index[min(slow, len(prices) - 1)]
    window_prices = prices[prices.index >= start]
    window_positions = positions[positions.index >= start]

    strategy = backtest_equity(window_prices, window_positions, cost_bps)
    buy_hold = window_prices / window_prices.iloc[0]

    trades = int((window_positions.diff().abs() > 0).sum())
    time_in_market = float(window_positions.mean())

    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "as_of": as_of(window_prices),
        "window": {
            "start": str(window_prices.index.min().date()),
            "end": str(window_prices.index.max().date()),
            "trading_days": int(len(window_prices)),
        },
        "parameters": {
            "fast": fast,
            "slow": slow,
            "cost_bps": cost_bps,
            "rule": "long when SMA(fast) > SMA(slow), else flat; signal lagged 1 day",
        },
        "strategy": summarize_prices(strategy),
        "buy_and_hold": summarize_prices(buy_hold),
        "trades": trades,
        "time_in_market": _round(time_in_market),
        "caveats": CAVEATS,
    }


INSTRUCTION = """
## Backtesting

For "how would X have done" questions, call `backtest_sma_cross`. Always
relay the result's caveats -- at minimum that the result is in-sample and not
predictive -- and always show buy-and-hold next to the strategy. If the user
proposes parameters, run exactly those rather than tuning for a better-looking
outcome.
""".strip()

TOOLS = [backtest_sma_cross]

"""
Technical snapshot skill: indicator readings computed locally from full daily
history, rather than fetched pre-chewed or estimated by the model.
"""

from __future__ import annotations

import logging
from typing import Any

from ._market_data import MarketDataError, as_of, get_history
from ._quant import _round, annualized_vol, rsi, simple_returns, sma

logger = logging.getLogger(__name__)


def technical_snapshot(ticker: str) -> dict[str, Any]:
    """Compute a technical snapshot for one stock or ETF from daily history.

    Reports: last close; 50- and 200-day simple moving averages and where
    price sits relative to them; the most recent golden/death cross and how
    long ago it was; Wilder RSI(14); 12-month and 1-month momentum; distance
    from the 52-week high and low; and 30-day realized volatility. Computed
    locally from ~2 years of daily adjusted closes. These describe the past
    -- never present them as predictive signals.

    Args:
        ticker: One ticker symbol, e.g. 'NVDA'.

    Returns:
        A dict of indicator readings with the as-of date, or an 'error'
        explaining what went wrong.
    """
    try:
        prices = get_history(ticker, years=2.0)
    except MarketDataError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - tool results must not raise
        logger.exception("technical_snapshot failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    if len(prices) < 210:
        return {
            "status": "error",
            "error": (
                f"Only {len(prices)} trading days of history for "
                f"'{ticker.upper()}'; the 200-day average needs at least 210."
            ),
        }

    close = float(prices.iloc[-1])
    sma50_series = sma(prices, 50)
    sma200_series = sma(prices, 200)
    sma50 = float(sma50_series.iloc[-1])
    sma200 = float(sma200_series.iloc[-1])

    # Most recent cross of the 50 over/under the 200.
    above = (sma50_series > sma200_series).dropna()
    cross_kind = None
    cross_days_ago = None
    changes = above[above != above.shift(1)].dropna()
    if len(changes) > 1:  # first entry is just the series start, not a cross
        last_change_date = changes.index[-1]
        cross_kind = "golden_cross" if bool(changes.iloc[-1]) else "death_cross"
        cross_days_ago = int((above.index[-1] - last_change_date).days)

    year_window = prices.iloc[-252:]
    high_52wk = float(year_window.max())
    low_52wk = float(year_window.min())

    returns = simple_returns(prices)

    def momentum(days: int) -> float | None:
        if len(prices) <= days:
            return None
        return _round(float(prices.iloc[-1] / prices.iloc[-1 - days] - 1))

    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "as_of": as_of(prices),
        "note": (
            "End-of-day data; historical description, not a trading signal or "
            "forecast."
        ),
        "close": _round(close, 2),
        "sma_50": _round(sma50, 2),
        "sma_200": _round(sma200, 2),
        "close_vs_sma50": _round(close / sma50 - 1),
        "close_vs_sma200": _round(close / sma200 - 1),
        "last_cross": cross_kind,
        "last_cross_days_ago": cross_days_ago,
        "rsi_14": _round(rsi(prices, 14), 1),
        "momentum_12m": momentum(252),
        "momentum_1m": momentum(21),
        "high_52wk": _round(high_52wk, 2),
        "low_52wk": _round(low_52wk, 2),
        "off_52wk_high": _round(close / high_52wk - 1),
        "above_52wk_low": _round(close / low_52wk - 1),
        "realized_vol_30d": _round(annualized_vol(returns.iloc[-30:])),
    }


INSTRUCTION = """
## Technical snapshot

For questions about a ticker's trend, moving averages, RSI, momentum, 52-week
range, or recent volatility, call `technical_snapshot` -- never estimate an
indicator or compute one from a handful of fetched prices. Describe readings
neutrally (e.g. "RSI 72, a level conventionally described as overbought");
do not turn them into buy/sell language.
""".strip()

TOOLS = [technical_snapshot]

"""
Portfolio analytics skill: risk/return statistics computed locally over full
daily history, so the model reports exact numbers instead of estimating.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ._market_data import MarketDataError, as_of, get_history_many
from ._quant import (
    _round,
    correlation_matrix,
    simple_returns,
    summarize_prices,
)

logger = logging.getLogger(__name__)


def analyze_portfolio(
    tickers: list[str],
    weights: list[float] | None = None,
    years: float = 3.0,
) -> dict[str, Any]:
    """Compute risk and return statistics for a portfolio of stocks or ETFs.

    Computes, per ticker and for the weighted portfolio: CAGR, annualized
    volatility, Sharpe and Sortino ratios (risk-free rate 0), maximum
    drawdown, and total return, plus the pairwise return correlation matrix.
    All figures are computed locally from daily adjusted closes -- report
    them as historical description, never as a forecast.

    Args:
        tickers: 1-20 ticker symbols, e.g. ['NVDA', 'MSFT', 'SPY'].
        weights: Optional portfolio weights matching the ticker order. They
            must sum to roughly 1.0. Omitted means equal-weight.
        years: Lookback window in years, 0.5 to 10. Defaults to 3.

    Returns:
        A dict with 'as_of' date, 'per_ticker' stats, 'portfolio' stats
        (weights applied, rebalanced daily), and a 'correlation' matrix.
        On bad input, a dict with an 'error' explaining what to fix.
    """
    try:
        years = float(min(max(years, 0.5), 10.0))
        tickers = [str(t) for t in tickers]

        if weights is not None:
            if len(weights) != len(tickers):
                return {
                    "status": "error",
                    "error": f"{len(tickers)} tickers but {len(weights)} weights.",
                }
            total = sum(weights)
            if not 0.98 <= total <= 1.02:
                return {
                    "status": "error",
                    "error": f"Weights sum to {total:.3f}; they must sum to 1.0.",
                }
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(tickers)] * len(tickers)

        closes = get_history_many(tickers, years=years)
    except MarketDataError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - tool results must not raise
        logger.exception("analyze_portfolio failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    returns = closes.pct_change().dropna()

    # Daily-rebalanced portfolio: weights held constant each day. State the
    # convention in the output so the number is interpretable.
    weight_series = pd.Series(dict(zip(closes.columns, weights)))
    portfolio_returns = (returns * weight_series).sum(axis=1)
    portfolio_equity = (1.0 + portfolio_returns).cumprod()

    correlation = correlation_matrix(returns)

    return {
        "status": "ok",
        "as_of": as_of(closes),
        "window_years": years,
        "conventions": (
            "Daily adjusted closes; 252-day annualization; CAGR is geometric; "
            "Sharpe/Sortino use arithmetic mean excess return with risk-free "
            "rate 0; portfolio is rebalanced daily to fixed weights. "
            "Historical description only, not a forecast."
        ),
        "per_ticker": {
            symbol: summarize_prices(closes[symbol]) for symbol in closes.columns
        },
        "portfolio": {
            "weights": {s: _round(w) for s, w in weight_series.items()},
            **summarize_prices(portfolio_equity),
        },
        "correlation": {
            row: {col: _round(float(correlation.loc[row, col])) for col in correlation.columns}
            for row in correlation.index
        },
    }


INSTRUCTION = """
## Portfolio analytics

For any question about portfolio risk, return, Sharpe, drawdown, volatility,
correlation, or diversification across specific tickers, call
`analyze_portfolio` instead of estimating. If the user's corpus notes define
their own preferred conventions, mention any difference from this tool's
conventions (reported in the result) rather than silently mixing them.
""".strip()

TOOLS = [analyze_portfolio]

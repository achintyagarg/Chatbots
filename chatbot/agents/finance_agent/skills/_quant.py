"""
Pure quant math over pandas series. No I/O, no network, no tool plumbing --
each function takes prices/returns in and hands statistics back, so the unit
tests can exercise every convention against hand-computed answers.

Conventions (stated here once; every tool that reports these numbers relies
on them and the docstrings tell the model to disclose them on request):

- Returns are daily simple returns on (auto-)adjusted closes.
- Annualization uses 252 trading days.
- Annualized return is CAGR (geometric), not mean*252.
- Sharpe / Sortino use arithmetic mean excess return over annualized vol /
  downside vol, with risk-free rate 0 unless supplied.
- Volatility uses the sample standard deviation (ddof=1).
- Max drawdown is the worst peak-to-trough decline of the price path.
- RSI is Wilder's smoothing (alpha = 1/period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def cagr(prices: pd.Series) -> float | None:
    """Geometric annualized return over the whole series."""
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return None
    total = prices.iloc[-1] / prices.iloc[0]
    if total <= 0:
        return None
    years = (len(prices) - 1) / TRADING_DAYS
    if years <= 0:
        return None
    return float(total ** (1 / years) - 1)


def annualized_vol(returns: pd.Series) -> float | None:
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


# Below this, volatility is floating-point residue of an arithmetically
# constant series, not a real risk estimate. Dividing by it would report a
# Sharpe of ~1e14, which is exactly the kind of authoritative-looking nonsense
# a ratio must never emit.
_VOL_EPSILON = 1e-9


def sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float | None:
    """Arithmetic-mean Sharpe. None when vol is ~zero (undefined, not 0)."""
    vol = annualized_vol(returns)
    if vol is None or vol < _VOL_EPSILON:
        return None
    excess = float(returns.mean()) * TRADING_DAYS - risk_free_rate
    return excess / vol


def sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float | None:
    """Like Sharpe but penalizing only downside deviation."""
    if len(returns) < 2:
        return None
    downside = returns[returns < 0]
    if len(downside) < 2:
        return None  # no meaningful downside estimate, not "infinitely good"
    downside_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if downside_vol < _VOL_EPSILON:
        return None
    excess = float(returns.mean()) * TRADING_DAYS - risk_free_rate
    return excess / downside_vol


def max_drawdown(prices: pd.Series) -> float | None:
    """Worst peak-to-trough decline, as a negative fraction (-0.25 = -25%)."""
    if len(prices) < 2:
        return None
    drawdowns = prices / prices.cummax() - 1.0
    return float(drawdowns.min())


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def rsi(prices: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI of the most recent bar."""
    if len(prices) <= period:
        return None
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def sma_cross_positions(
    prices: pd.Series, fast: int, slow: int
) -> pd.Series:
    """
    Long-or-flat positions for an SMA crossover, **lagged one bar**.

    The signal computed from bar T's close can only be traded at T+1; using it
    at T is lookahead bias, the classic way backtests flatter themselves. The
    first `slow` bars are flat because the slow SMA does not exist yet.
    """
    signal = (sma(prices, fast) > sma(prices, slow)).astype(float)
    return signal.shift(1).fillna(0.0)


def backtest_equity(
    prices: pd.Series, positions: pd.Series, cost_bps: float = 0.0
) -> pd.Series:
    """
    Equity curve for a position series, charging ``cost_bps`` (one-way, in
    basis points of traded notional) on every position change -- including the
    entry and the final state's implied history. Costs make crossover systems
    noticeably worse; a backtest without them is an advertisement.
    """
    returns = prices.pct_change().fillna(0.0)
    strategy_returns = positions * returns
    trades = positions.diff().abs().fillna(positions.abs())
    strategy_returns = strategy_returns - trades * (cost_bps / 10_000.0)
    return (1.0 + strategy_returns).cumprod()


def summarize_prices(prices: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """The standard per-series block every tool reports."""
    returns = simple_returns(prices)
    return {
        "cagr": _round(cagr(prices)),
        "annualized_vol": _round(annualized_vol(returns)),
        "sharpe": _round(sharpe(returns, risk_free_rate)),
        "sortino": _round(sortino(returns, risk_free_rate)),
        "max_drawdown": _round(max_drawdown(prices)),
        "total_return": _round(
            float(prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) >= 2 else None
        ),
        "observations": int(len(prices)),
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)

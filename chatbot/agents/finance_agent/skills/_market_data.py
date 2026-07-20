"""
Price history for the quant skills: yfinance as a *library*, with a local
cache. Deliberately not an MCP tool -- a backtest needs years of OHLCV, and
shuttling thousands of rows through model context as tool output would be
slow, expensive, and lossy. Skills fetch here, compute locally, and return
summary statistics; the model reasons over results, never raw bars.

Cache: one CSV of daily adjusted closes per symbol under data/price_cache/,
refreshed when older than a day. Quotes are end-of-day; the interactive MCP
tools are the source for "right now" numbers.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "price_cache"
CACHE_TTL_SECONDS = 24 * 3600
HISTORY_PERIOD = "10y"  # one fetch covers every skill's longest lookback

_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,9}([.-][A-Z0-9]{1,4})?$")


class MarketDataError(Exception):
    """Raised with a message safe to surface in a tool result."""


def _validate(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not _TICKER.match(symbol):
        raise MarketDataError(f"'{symbol}' does not look like a ticker symbol.")
    return symbol


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _read_cache(symbol: str) -> pd.Series | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        series = frame["close"].dropna()
        series.name = symbol
        return series if len(series) else None
    except (ValueError, KeyError, OSError):
        logger.exception("Unreadable cache for %s; refetching", symbol)
        return None


def _write_cache(symbol: str, series: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_frame().to_csv(_cache_path(symbol))


def _fetch(symbol: str) -> pd.Series:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
    if frame is None or frame.empty or "Close" not in frame:
        raise MarketDataError(
            f"No price history returned for '{symbol}'. Check the symbol; "
            "Yahoo uses suffixes for non-US listings (e.g. '7203.T')."
        )
    series = frame["Close"].dropna()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series.name = symbol
    return series


def get_history(symbol: str, years: float = 3.0) -> pd.Series:
    """
    Daily adjusted closes for ``symbol`` covering the last ``years``.

    Serves from the local cache when fresh; otherwise one network fetch of the
    full 10y window so subsequent skills hit the cache regardless of lookback.
    """
    symbol = _validate(symbol)

    series = _read_cache(symbol)
    if series is None:
        series = _fetch(symbol)
        _write_cache(symbol, series)

    cutoff = series.index.max() - pd.Timedelta(days=round(years * 365.25))
    window = series[series.index >= cutoff]
    if len(window) < 2:
        raise MarketDataError(
            f"Only {len(window)} data point(s) for '{symbol}' in the last "
            f"{years:g} year(s) -- not enough to compute statistics."
        )
    return window


def get_history_many(symbols: list[str], years: float = 3.0) -> pd.DataFrame:
    """
    Aligned close frame for several symbols (inner join on dates, so every row
    has a price for every symbol -- correlation on misaligned dates is noise).
    """
    if not symbols:
        raise MarketDataError("No ticker symbols were given.")
    if len(symbols) > 20:
        raise MarketDataError("At most 20 symbols per call.")

    columns = {}
    for symbol in symbols:
        columns[_validate(symbol)] = get_history(symbol, years=years)

    frame = pd.DataFrame(columns).dropna()
    if len(frame) < 2:
        raise MarketDataError(
            "The symbols share fewer than 2 overlapping trading days; "
            "cannot compute joint statistics."
        )
    return frame


def as_of(series_or_frame) -> str:
    """Last date in the data, for the model to report as the as-of date."""
    return str(series_or_frame.index.max().date())

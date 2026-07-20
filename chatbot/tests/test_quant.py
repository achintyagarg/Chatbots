"""
Known-answer tests for the quant math.

Every series here is constructed so the right answer is known by hand, not by
re-running the code under test. If a convention changes (annualization,
ddof, Wilder smoothing), a number stops matching and the test says which one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from finance_agent.skills import _market_data  # noqa: E402
from finance_agent.skills._quant import (  # noqa: E402
    TRADING_DAYS,
    annualized_vol,
    backtest_equity,
    cagr,
    max_drawdown,
    rsi,
    sharpe,
    simple_returns,
    sma_cross_positions,
    sortino,
    summarize_prices,
)


def series(values, start="2024-01-01"):
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=index)


def constant_growth(rate: float, periods: int) -> pd.Series:
    return series([100.0 * (1.0 + rate) ** i for i in range(periods)])


class TestReturnsAndCagr:
    def test_simple_returns_known_values(self):
        prices = series([100, 110, 99])
        returns = simple_returns(prices)
        assert returns.iloc[0] == pytest.approx(0.10)
        assert returns.iloc[1] == pytest.approx(-0.10)

    def test_cagr_of_constant_growth_is_the_rate_annualized(self):
        # 1% per day for exactly one trading year: CAGR = 1.01^252 - 1.
        prices = constant_growth(0.01, TRADING_DAYS + 1)
        assert cagr(prices) == pytest.approx(1.01**TRADING_DAYS - 1, rel=1e-9)

    def test_cagr_flat_series_is_zero(self):
        assert cagr(series([100] * 50)) == pytest.approx(0.0)

    def test_cagr_needs_two_points(self):
        assert cagr(series([100])) is None


class TestVolAndRatios:
    def test_constant_growth_has_zero_vol_and_undefined_sharpe(self):
        prices = constant_growth(0.001, 100)
        returns = simple_returns(prices)
        assert annualized_vol(returns) == pytest.approx(0.0)
        # Zero vol makes Sharpe undefined; None, not infinity or zero.
        assert sharpe(returns) is None

    def test_annualized_vol_alternating_series(self):
        # Returns alternate +1%/-1%: sample std is known in closed form.
        returns = pd.Series([0.01, -0.01] * 50)
        expected = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        assert annualized_vol(returns) == pytest.approx(float(expected))

    def test_sharpe_sign_follows_mean(self):
        up = pd.Series([0.02, -0.01] * 50)
        down = pd.Series([-0.02, 0.01] * 50)
        assert sharpe(up) > 0
        assert sharpe(down) < 0
        assert sharpe(up) == pytest.approx(-sharpe(down))

    def test_sortino_none_without_downside(self):
        returns = pd.Series([0.01] * 30 + [0.02] * 30)
        assert sortino(returns) is None

    def test_sortino_exceeds_sharpe_for_upside_skew(self):
        # Rare, varied small losses among frequent gains: downside vol is real
        # but far below total vol. (The losses must differ from each other --
        # identical losses have zero downside deviation and Sortino is None.)
        returns = pd.Series(([0.01] * 8 + [-0.004, -0.008]) * 10)
        assert sortino(returns) > sharpe(returns)


class TestDrawdown:
    def test_known_path(self):
        # Peak 120, trough 90: max drawdown is 90/120 - 1 = -25%.
        prices = series([100, 120, 90, 130])
        assert max_drawdown(prices) == pytest.approx(-0.25)

    def test_monotonic_rise_has_zero_drawdown(self):
        assert max_drawdown(constant_growth(0.01, 50)) == pytest.approx(0.0)

    def test_drawdown_measured_from_running_peak(self):
        # Second, deeper trough measured from second, higher peak.
        prices = series([100, 110, 100, 150, 105, 160])
        assert max_drawdown(prices) == pytest.approx(105 / 150 - 1)


class TestRsi:
    def test_all_gains_is_100(self):
        assert rsi(constant_growth(0.01, 40)) == pytest.approx(100.0)

    def test_needs_more_than_period(self):
        assert rsi(series([100, 101, 102]), period=14) is None

    def test_symmetric_chop_is_near_50(self):
        prices = series([100 + (1 if i % 2 else 0) for i in range(60)])
        value = rsi(prices)
        assert 40 <= value <= 60


class TestSmaCrossBacktest:
    def make_v_shape(self):
        # 60 flat days, then a strong linear rise: the fast SMA must cross
        # above the slow SMA during the rise, on a date we can find directly
        # from the SMAs themselves.
        return series([100.0] * 60 + [100.0 + 2 * i for i in range(1, 61)])

    def test_positions_are_lagged_one_bar(self):
        prices = self.make_v_shape()
        fast, slow = 5, 20
        positions = sma_cross_positions(prices, fast, slow)

        raw_signal = (
            prices.rolling(fast).mean() > prices.rolling(slow).mean()
        ).astype(float)
        first_signal_day = raw_signal[raw_signal > 0].index[0]
        first_position_day = positions[positions > 0].index[0]

        # The position starts exactly one trading day AFTER the signal --
        # trading on the signal day itself would be lookahead bias.
        gap = positions.index.get_loc(first_position_day) - raw_signal.index.get_loc(
            first_signal_day
        )
        assert gap == 1

    def test_flat_before_slow_sma_exists(self):
        prices = self.make_v_shape()
        positions = sma_cross_positions(prices, 5, 20)
        assert (positions.iloc[:20] == 0).all()

    def test_costs_only_reduce_equity(self):
        prices = self.make_v_shape()
        positions = sma_cross_positions(prices, 5, 20)
        free = backtest_equity(prices, positions, cost_bps=0.0)
        costly = backtest_equity(prices, positions, cost_bps=25.0)
        assert costly.iloc[-1] < free.iloc[-1]

    def test_always_flat_equity_stays_at_one(self):
        prices = self.make_v_shape()
        flat = pd.Series(0.0, index=prices.index)
        equity = backtest_equity(prices, flat, cost_bps=10.0)
        assert equity.iloc[-1] == pytest.approx(1.0)

    def test_always_long_no_cost_tracks_buy_and_hold(self):
        prices = self.make_v_shape()
        long = pd.Series(1.0, index=prices.index)
        equity = backtest_equity(prices, long, cost_bps=0.0)
        buy_hold_return = prices.iloc[-1] / prices.iloc[0]
        # One unit of cost-free exposure from day one replicates buy-and-hold
        # except the entry trade: equity compounds from the first return on.
        assert equity.iloc[-1] == pytest.approx(buy_hold_return, rel=1e-9)


class TestSummarize:
    def test_block_is_json_safe_and_rounded(self):
        block = summarize_prices(constant_growth(0.001, 300))
        assert block["observations"] == 300
        assert block["sharpe"] is None  # zero vol
        for value in block.values():
            assert value is None or isinstance(value, (int, float))


class TestMarketDataCache:
    def test_cache_hit_never_touches_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_market_data, "CACHE_DIR", tmp_path)

        cached = constant_growth(0.001, 400)
        _market_data._write_cache("TEST", cached)

        def explode(*args, **kwargs):
            raise AssertionError("network fetch called despite fresh cache")

        monkeypatch.setattr(_market_data, "_fetch", explode)
        result = _market_data.get_history("TEST", years=1.0)
        assert len(result) > 200

    def test_stale_cache_refetches(self, monkeypatch, tmp_path):
        import os
        import time

        monkeypatch.setattr(_market_data, "CACHE_DIR", tmp_path)
        _market_data._write_cache("TEST", constant_growth(0.001, 400))

        stale = time.time() - _market_data.CACHE_TTL_SECONDS - 60
        os.utime(_market_data._cache_path("TEST"), (stale, stale))

        fresh = constant_growth(0.002, 300)
        monkeypatch.setattr(_market_data, "_fetch", lambda symbol: fresh)
        result = _market_data.get_history("TEST", years=1.0)
        assert result.iloc[-1] == pytest.approx(fresh.iloc[-1])

    def test_rejects_garbage_symbol(self):
        with pytest.raises(_market_data.MarketDataError):
            _market_data.get_history("not a ticker!!", years=1.0)

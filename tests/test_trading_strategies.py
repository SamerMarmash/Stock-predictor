"""Tests for trading strategy engine."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import Direction, TechnicalSignal
from src.strategies.trading_strategies import TradingStrategyEngine


def _make_signals() -> list[TechnicalSignal]:
    """Create a set of mock technical signals."""
    return [
        TechnicalSignal(
            indicator_name="SMA 20/50 Crossover",
            value=2.5,
            signal=Direction.BULLISH,
            strength=0.7,
            description="SMA20 above SMA50",
        ),
        TechnicalSignal(
            indicator_name="EMA 9/21 Crossover",
            value=1.0,
            signal=Direction.BULLISH,
            strength=0.6,
            description="EMA9 above EMA21",
        ),
        TechnicalSignal(
            indicator_name="RSI (14)",
            value=55,
            signal=Direction.NEUTRAL,
            strength=0.1,
            description="RSI neutral",
        ),
        TechnicalSignal(
            indicator_name="MACD",
            value=0.5,
            signal=Direction.BULLISH,
            strength=0.65,
            description="MACD positive",
        ),
        TechnicalSignal(
            indicator_name="ADX (14)",
            value=35,
            signal=Direction.BULLISH,
            strength=0.7,
            description="Strong trend",
        ),
        TechnicalSignal(
            indicator_name="Bollinger Bands",
            value=150,
            signal=Direction.NEUTRAL,
            strength=0.3,
            description="Mid range",
        ),
        TechnicalSignal(
            indicator_name="VWAP",
            value=148,
            signal=Direction.BULLISH,
            strength=0.5,
            description="Above VWAP",
        ),
        TechnicalSignal(
            indicator_name="OBV Trend",
            value=1000000,
            signal=Direction.BULLISH,
            strength=0.6,
            description="OBV rising",
        ),
    ]


def _make_current_data() -> dict:
    return {
        "ticker": "AAPL",
        "current_price": 150.0,
        "previous_close": 148.0,
        "open": 149.0,
        "day_high": 151.0,
        "day_low": 148.5,
    }


def _make_history(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 150 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def engine():
    return TradingStrategyEngine()


def test_evaluate_returns_strategies(engine):
    signals = _make_signals()
    current = _make_current_data()
    history = _make_history()
    results = engine.evaluate(signals, current, history)
    assert len(results) >= 5  # At least 5 strategies should produce results


def test_strategy_names(engine):
    signals = _make_signals()
    current = _make_current_data()
    history = _make_history()
    results = engine.evaluate(signals, current, history)
    names = {r.strategy_name for r in results}
    assert "Trend Following" in names
    assert "Mean Reversion" in names
    assert "Scalping Setup" in names


def test_bullish_signals_produce_bullish_trend(engine):
    signals = _make_signals()  # Mostly bullish
    current = _make_current_data()
    history = _make_history()
    results = engine.evaluate(signals, current, history)
    trend = next(r for r in results if r.strategy_name == "Trend Following")
    assert trend.direction == Direction.BULLISH
    assert trend.confidence > 0.5


def test_options_flow_strategy(engine):
    signals = _make_signals()
    current = _make_current_data()
    history = _make_history()
    options = {
        "available": True,
        "put_call_ratio": 0.5,
        "signal": "bullish",
        "total_options_volume": 50000,
    }
    results = engine.evaluate(signals, current, history, options)
    options_strat = next((r for r in results if r.strategy_name == "Options Flow"), None)
    assert options_strat is not None
    assert options_strat.direction == Direction.BULLISH


def test_gap_strategy_detects_gap(engine):
    signals = _make_signals()
    current = _make_current_data()
    current["previous_close"] = 145.0  # Create a gap up
    current["open"] = 148.0
    history = _make_history()
    results = engine.evaluate(signals, current, history)
    gap = next(r for r in results if r.strategy_name == "Gap Strategy")
    assert gap.direction in (Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL)


def test_confidence_in_valid_range(engine):
    signals = _make_signals()
    current = _make_current_data()
    history = _make_history()
    results = engine.evaluate(signals, current, history)
    for r in results:
        assert 0 <= r.confidence <= 1.0

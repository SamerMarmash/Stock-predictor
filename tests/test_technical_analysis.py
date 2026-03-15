"""Tests for technical analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import Direction
from src.strategies.technical_analysis import TechnicalAnalyzer


def _make_price_data(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    if trend == "up":
        base = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    elif trend == "down":
        base = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    else:
        base = 100 + np.cumsum(np.random.randn(n) * 0.3)

    high = base + np.abs(np.random.randn(n)) * 0.5
    low = base - np.abs(np.random.randn(n)) * 0.5
    volume = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {"Open": base, "High": high, "Low": low, "Close": base, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def analyzer():
    return TechnicalAnalyzer()


def test_analyze_returns_signals(analyzer):
    df = _make_price_data(100)
    signals = analyzer.analyze(df)
    assert len(signals) > 0
    for s in signals:
        assert s.indicator_name
        assert s.signal in (Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL)
        assert 0 <= s.strength <= 1


def test_analyze_empty_df(analyzer):
    df = pd.DataFrame()
    signals = analyzer.analyze(df)
    assert signals == []


def test_analyze_too_short(analyzer):
    df = _make_price_data(5)
    signals = analyzer.analyze(df)
    assert signals == []


def test_uptrend_has_bullish_signals(analyzer):
    df = _make_price_data(100, trend="up")
    signals = analyzer.analyze(df)
    bullish_count = sum(1 for s in signals if s.signal == Direction.BULLISH)
    bearish_count = sum(1 for s in signals if s.signal == Direction.BEARISH)
    # In a clear uptrend, we expect more bullish signals
    assert bullish_count > 0


def test_downtrend_has_bearish_signals(analyzer):
    df = _make_price_data(100, trend="down")
    signals = analyzer.analyze(df)
    bearish_count = sum(1 for s in signals if s.signal == Direction.BEARISH)
    assert bearish_count > 0


def test_indicator_names_present(analyzer):
    df = _make_price_data(100)
    signals = analyzer.analyze(df)
    names = {s.indicator_name for s in signals}
    # Should include key indicators
    assert "RSI (14)" in names
    assert "MACD" in names
    assert "Bollinger Bands" in names

"""Tests for domain models."""

from datetime import datetime

from src.core.models import (
    Direction,
    ExpertValidation,
    IntelligenceItem,
    SourceType,
    StockPrediction,
    StrategySignal,
    Subscription,
    TechnicalSignal,
    TimeHorizon,
)


def test_intelligence_item_defaults():
    item = IntelligenceItem(
        source_type=SourceType.NEWS,
        source_name="Test",
        title="Test Title",
        content="Test content",
    )
    assert item.sentiment_score == 0.0
    assert item.relevance_score == 0.0
    assert item.credibility_score == 0.5
    assert item.url == ""


def test_technical_signal():
    signal = TechnicalSignal(
        indicator_name="RSI",
        value=72.5,
        signal=Direction.BEARISH,
        strength=0.8,
        description="Overbought",
    )
    assert signal.indicator_name == "RSI"
    assert signal.signal == Direction.BEARISH
    assert signal.strength == 0.8


def test_strategy_signal():
    signal = StrategySignal(
        strategy_name="Trend Following",
        direction=Direction.BULLISH,
        confidence=0.75,
        reasoning="All MAs aligned upward",
    )
    assert signal.confidence == 0.75
    assert signal.direction == Direction.BULLISH


def test_stock_prediction():
    prediction = StockPrediction(
        ticker="AAPL",
        company_name="Apple Inc.",
        direction=Direction.BULLISH,
        predicted_price_change_pct=2.5,
        confidence_pct=65.0,
        current_price=185.0,
        predicted_price=189.63,
        time_horizon=TimeHorizon.DAILY,
        key_drivers=["Strong earnings", "Positive sentiment"],
        intelligence_summary="Multiple bullish signals",
        strategy_signals=[],
        expert_validations=[],
        source_count=42,
        bull_case="Strong fundamentals",
        bear_case="Valuation concerns",
    )
    assert prediction.ticker == "AAPL"
    assert prediction.confidence_pct == 65.0
    assert len(prediction.key_drivers) == 2


def test_expert_validation():
    val = ExpertValidation(
        expert_name="Warren Buffett",
        agrees_with_prediction=True,
        expert_direction=Direction.BULLISH,
        credibility_rating=0.9,
        summary="Increased AAPL position in Q3",
    )
    assert val.credibility_rating == 0.9
    assert val.agrees_with_prediction is True


def test_subscription():
    sub = Subscription(ticker="TSLA", confidence_threshold=70)
    assert sub.ticker == "TSLA"
    assert sub.confidence_threshold == 70
    assert sub.user_id == "default"
    assert sub.notify is True


def test_direction_enum():
    assert Direction.BULLISH.value == "bullish"
    assert Direction.BEARISH.value == "bearish"
    assert Direction.NEUTRAL.value == "neutral"


def test_source_type_enum():
    assert SourceType.NEWS.value == "news"
    assert SourceType.SOCIAL_MEDIA.value == "social_media"
    assert SourceType.YOUTUBE.value == "youtube"
    assert SourceType.EXPERT_OPINION.value == "expert_opinion"

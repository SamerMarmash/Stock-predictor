"""Domain models used across the application."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TimeHorizon(str, Enum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"


class SourceType(str, Enum):
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    SEC_FILING = "sec_filing"
    ANALYST_REPORT = "analyst_report"
    EXPERT_OPINION = "expert_opinion"
    TECHNICAL_INDICATOR = "technical_indicator"
    RSS_FEED = "rss_feed"
    WEB_SCRAPE = "web_scrape"


class IntelligenceItem(BaseModel):
    """A single piece of market intelligence from any source."""

    source_type: SourceType
    source_name: str
    title: str
    content: str
    url: str = ""
    author: str = ""
    published_at: datetime | None = None
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TechnicalSignal(BaseModel):
    """Result from a technical analysis indicator."""

    indicator_name: str
    value: float
    signal: Direction
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""


class StrategySignal(BaseModel):
    """Result from applying a trading strategy."""

    strategy_name: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_indicators: list[TechnicalSignal] = []


class ExpertValidation(BaseModel):
    """Fact-check result from a reputable trader/analyst."""

    expert_name: str
    source_url: str = ""
    agrees_with_prediction: bool
    expert_direction: Direction
    credibility_rating: float = Field(ge=0.0, le=1.0)
    summary: str


class StockPrediction(BaseModel):
    """The final prediction output for a stock."""

    ticker: str
    company_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    direction: Direction
    predicted_price_change_pct: float
    confidence_pct: float = Field(ge=0.0, le=100.0)
    current_price: float
    predicted_price: float
    time_horizon: TimeHorizon

    # Supporting evidence
    key_drivers: list[str]
    intelligence_summary: str
    strategy_signals: list[StrategySignal]
    expert_validations: list[ExpertValidation]
    source_count: int
    bull_case: str
    bear_case: str

    # Metadata
    model_used: str = ""
    generation_time_seconds: float = 0.0


class Subscription(BaseModel):
    """User subscription to a stock ticker."""

    ticker: str
    user_id: str = "default"
    confidence_threshold: int = 60
    notify: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

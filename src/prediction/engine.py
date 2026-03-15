"""AI prediction engine - the brain of the system.

Combines technical analysis, multi-source intelligence, trading strategies,
and expert validation to produce a final prediction with confidence percentage.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import structlog

from src.core.config import get_settings
from src.core.llm_client import LLMClient
from src.core.market_data import MarketDataProvider
from src.core.models import (
    Direction,
    ExpertValidation,
    IntelligenceItem,
    StockPrediction,
    StrategySignal,
    TechnicalSignal,
    TimeHorizon,
)
from src.intelligence.aggregator import IntelligenceAggregator
from src.prediction.fact_checker import FactChecker
from src.strategies.technical_analysis import TechnicalAnalyzer
from src.strategies.trading_strategies import TradingStrategyEngine

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are an elite quantitative analyst and professional day trader with decades of
experience. You have deep knowledge of:

- Technical analysis: candlestick patterns, chart patterns, indicators (RSI, MACD, Bollinger Bands,
  Fibonacci, Ichimoku, volume profile, order flow)
- Fundamental analysis: earnings, revenue growth, PE ratios, sector rotation
- Sentiment analysis: social media sentiment, news impact, fear/greed indicators
- Market microstructure: bid-ask spreads, dark pool activity, options flow
- Trading strategies: trend following, mean reversion, breakout, scalping, pairs trading,
  momentum, gap trading, VWAP strategies
- Risk management: position sizing, stop-loss placement, risk-reward ratios
- Behavioral finance: market psychology, institutional vs retail flow
- Macroeconomics: Fed policy, interest rates, inflation, GDP impact on equities

Your job is to synthesize ALL provided data (technical signals, news intelligence, social sentiment,
expert opinions, and trading strategy results) into a single, well-reasoned stock prediction.

You must be honest about uncertainty and never overstate confidence. Consider contrarian viewpoints.
Factor in the current market regime (bull, bear, or range-bound).

IMPORTANT: You must respond with valid JSON only. No explanatory text outside the JSON."""

PREDICTION_PROMPT_TEMPLATE = """Analyze the following data for {ticker} ({company_name}) and provide your prediction.

## Current Market Data
{market_data}

## Technical Analysis Signals
{technical_signals}

## Trading Strategy Results
{strategy_results}

## Intelligence Report ({intel_count} sources)
{intelligence_summary}

## Expert/Analyst Validations
{expert_validations}

## Options Flow
{options_data}

Based on ALL of the above, provide your prediction as JSON with this exact structure:
{{
    "direction": "bullish" or "bearish" or "neutral",
    "predicted_price_change_pct": <float, e.g. 2.5 for +2.5%>,
    "confidence_pct": <float 0-100, your honest confidence in this prediction>,
    "time_horizon": "intraday" or "daily" or "weekly",
    "key_drivers": ["driver1", "driver2", "driver3"],
    "intelligence_summary": "<2-3 sentence summary of what the intelligence tells us>",
    "bull_case": "<strongest bull argument in 1-2 sentences>",
    "bear_case": "<strongest bear argument in 1-2 sentences>",
    "predicted_price": <float, target price>
}}

Rules for confidence_pct:
- 80-100%: Extremely rare. Multiple strong signals all aligned, major catalyst confirmed.
- 60-79%: Strong setup with most signals aligned and credible intelligence support.
- 40-59%: Mixed signals or limited data. Be honest that the edge is small.
- 20-39%: Highly uncertain, conflicting signals. Acknowledge the uncertainty.
- 0-19%: Essentially a coin flip. No clear edge.

Be calibrated and honest. Most predictions should be in the 35-65% range."""


class PredictionEngine:
    """Orchestrates all components to produce a stock prediction."""

    def __init__(self):
        self.market = MarketDataProvider()
        self.technical = TechnicalAnalyzer()
        self.strategies = TradingStrategyEngine()
        self.intelligence = IntelligenceAggregator()
        self.fact_checker = FactChecker()
        self.llm = LLMClient()
        self.settings = get_settings()

    async def predict(self, ticker: str) -> StockPrediction:
        """Generate a full prediction for a stock ticker."""
        start = time.time()
        ticker = ticker.upper()
        logger.info("prediction_started", ticker=ticker)

        # Phase 1: Gather all data in parallel
        snapshot = await self.market.get_full_snapshot(ticker)
        current_data = snapshot["current"]
        company_name = current_data.get("company_name", ticker)
        history: pd.DataFrame = snapshot["history"]

        # Run technical analysis
        tech_signals = self.technical.analyze(history)

        # Run trading strategies
        strategy_signals = self.strategies.evaluate(
            tech_signals, current_data, history, snapshot.get("options")
        )

        # Gather intelligence from all sources
        intel_items = await self.intelligence.gather_all(ticker, company_name)

        # Fact-check against expert opinions
        expert_validations = await self.fact_checker.validate(
            ticker, company_name, tech_signals, intel_items
        )

        # Phase 2: Send everything to the AI for synthesis
        prediction = await self._synthesize(
            ticker=ticker,
            company_name=company_name,
            current_data=current_data,
            tech_signals=tech_signals,
            strategy_signals=strategy_signals,
            intel_items=intel_items,
            expert_validations=expert_validations,
            options_data=snapshot.get("options", {}),
        )

        prediction.generation_time_seconds = round(time.time() - start, 2)
        prediction.model_used = self.settings.llm_model
        prediction.strategy_signals = strategy_signals
        prediction.expert_validations = expert_validations
        prediction.source_count = len(intel_items)

        logger.info(
            "prediction_complete",
            ticker=ticker,
            direction=prediction.direction.value,
            confidence=prediction.confidence_pct,
            time=prediction.generation_time_seconds,
        )
        return prediction

    async def _synthesize(
        self,
        ticker: str,
        company_name: str,
        current_data: dict,
        tech_signals: list[TechnicalSignal],
        strategy_signals: list[StrategySignal],
        intel_items: list[IntelligenceItem],
        expert_validations: list[ExpertValidation],
        options_data: dict,
    ) -> StockPrediction:
        """Use the LLM to synthesize all data into a prediction."""

        # Format technical signals
        tech_text = "\n".join(
            f"- {s.indicator_name}: {s.signal.value} (strength: {s.strength:.0%}) - {s.description}"
            for s in tech_signals
        ) or "No technical signals available."

        # Format strategy results
        strategy_text = "\n".join(
            f"- {s.strategy_name}: {s.direction.value} (confidence: {s.confidence:.0%}) - {s.reasoning}"
            for s in strategy_signals
        ) or "No strategy signals available."

        # Summarize intelligence (take top 30 items by credibility)
        sorted_intel = sorted(intel_items, key=lambda x: x.credibility_score, reverse=True)[:30]
        intel_text = "\n".join(
            f"- [{i.source_type.value}] {i.source_name}: {i.title} (credibility: {i.credibility_score:.0%})"
            for i in sorted_intel
        ) or "No intelligence gathered."

        # Format expert validations
        expert_text = "\n".join(
            f"- {v.expert_name}: {v.expert_direction.value} (credibility: {v.credibility_rating:.0%}) - {v.summary}"
            for v in expert_validations
        ) or "No expert validations available."

        # Format options data
        if options_data and options_data.get("available"):
            options_text = (
                f"Put/Call Ratio: {options_data.get('put_call_ratio', 'N/A')}\n"
                f"Total Options Volume: {options_data.get('total_options_volume', 'N/A')}\n"
                f"Signal: {options_data.get('signal', 'N/A')}"
            )
        else:
            options_text = "Options data not available."

        # Format current market data
        market_text = json.dumps(
            {k: v for k, v in current_data.items() if not isinstance(v, (pd.DataFrame,))},
            indent=2,
            default=str,
        )

        prompt = PREDICTION_PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            market_data=market_text,
            technical_signals=tech_text,
            strategy_results=strategy_text,
            intel_count=len(intel_items),
            intelligence_summary=intel_text,
            expert_validations=expert_text,
            options_data=options_text,
        )

        try:
            result = await self.llm.complete_json(SYSTEM_PROMPT, prompt, max_tokens=2048)
        except Exception as e:
            logger.error("llm_synthesis_failed", error=str(e))
            # Fall back to strategy-based prediction
            return self._fallback_prediction(
                ticker, company_name, current_data, strategy_signals, intel_items
            )

        current_price = current_data.get("current_price", 0)
        predicted_price = result.get("predicted_price", current_price)
        change_pct = result.get("predicted_price_change_pct", 0)

        if predicted_price == current_price and change_pct != 0:
            predicted_price = current_price * (1 + change_pct / 100)

        return StockPrediction(
            ticker=ticker,
            company_name=company_name,
            direction=Direction(result.get("direction", "neutral")),
            predicted_price_change_pct=change_pct,
            confidence_pct=result.get("confidence_pct", 40),
            current_price=current_price,
            predicted_price=round(predicted_price, 2),
            time_horizon=TimeHorizon(result.get("time_horizon", "daily")),
            key_drivers=result.get("key_drivers", []),
            intelligence_summary=result.get("intelligence_summary", ""),
            strategy_signals=strategy_signals,
            expert_validations=[],
            source_count=len(intel_items),
            bull_case=result.get("bull_case", ""),
            bear_case=result.get("bear_case", ""),
        )

    def _fallback_prediction(
        self,
        ticker: str,
        company_name: str,
        current_data: dict,
        strategy_signals: list[StrategySignal],
        intel_items: list[IntelligenceItem],
    ) -> StockPrediction:
        """Produce a prediction from strategy signals when the LLM is unavailable."""
        bullish = sum(1 for s in strategy_signals if s.direction == Direction.BULLISH)
        bearish = sum(1 for s in strategy_signals if s.direction == Direction.BEARISH)
        total = len(strategy_signals) or 1

        if bullish > bearish:
            direction = Direction.BULLISH
            avg_conf = sum(s.confidence for s in strategy_signals if s.direction == Direction.BULLISH) / max(bullish, 1)
        elif bearish > bullish:
            direction = Direction.BEARISH
            avg_conf = sum(s.confidence for s in strategy_signals if s.direction == Direction.BEARISH) / max(bearish, 1)
        else:
            direction = Direction.NEUTRAL
            avg_conf = 0.4

        current_price = current_data.get("current_price", 0)
        change = 1.0 if direction == Direction.BULLISH else -1.0 if direction == Direction.BEARISH else 0
        predicted = current_price * (1 + change / 100)

        return StockPrediction(
            ticker=ticker,
            company_name=company_name,
            direction=direction,
            predicted_price_change_pct=change,
            confidence_pct=round(avg_conf * 100, 1),
            current_price=current_price,
            predicted_price=round(predicted, 2),
            time_horizon=TimeHorizon.DAILY,
            key_drivers=["Technical strategy consensus (LLM unavailable)"],
            intelligence_summary=f"Based on {len(strategy_signals)} trading strategies. {bullish} bullish, {bearish} bearish.",
            strategy_signals=strategy_signals,
            expert_validations=[],
            source_count=len(intel_items),
            bull_case="Strategy consensus leans bullish" if direction == Direction.BULLISH else "",
            bear_case="Strategy consensus leans bearish" if direction == Direction.BEARISH else "",
        )

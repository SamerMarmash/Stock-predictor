"""Fact-checking module that validates predictions against reputable sources.

Cross-references our signals with:
1. Known expert/analyst positions (Buffett, Burry, Cathie Wood, etc.)
2. Analyst consensus ratings
3. Institutional holdings changes (13F filings)
4. High-credibility news sources
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from bs4 import BeautifulSoup

from src.core.config import get_settings
from src.core.models import (
    Direction,
    ExpertValidation,
    IntelligenceItem,
    SourceType,
    TechnicalSignal,
)
from src.core.llm_client import LLMClient

logger = structlog.get_logger()

# Expert traders with historically verifiable track records
TRACKED_EXPERTS = [
    {"name": "Warren Buffett", "firm": "Berkshire Hathaway", "credibility": 0.90, "style": "value"},
    {"name": "Cathie Wood", "firm": "ARK Invest", "credibility": 0.65, "style": "growth"},
    {"name": "Michael Burry", "firm": "Scion Capital", "credibility": 0.75, "style": "contrarian"},
    {"name": "Ray Dalio", "firm": "Bridgewater", "credibility": 0.80, "style": "macro"},
    {"name": "Stanley Druckenmiller", "firm": "Duquesne", "credibility": 0.85, "style": "macro"},
    {"name": "Bill Ackman", "firm": "Pershing Square", "credibility": 0.70, "style": "activist"},
    {"name": "David Tepper", "firm": "Appaloosa", "credibility": 0.80, "style": "event-driven"},
    {"name": "Dan Ives", "firm": "Wedbush", "credibility": 0.65, "style": "tech analyst"},
    {"name": "Tom Lee", "firm": "Fundstrat", "credibility": 0.60, "style": "strategist"},
    {"name": "Jim Cramer", "firm": "CNBC", "credibility": 0.35, "style": "media"},
]

FACT_CHECK_SYSTEM_PROMPT = """You are a financial fact-checker. Given information about an expert's
recent statements or positions on a stock, determine their stance.

Respond with valid JSON only:
{
    "agrees_with_bullish": true or false,
    "expert_direction": "bullish" or "bearish" or "neutral",
    "summary": "1-2 sentence summary of the expert's position",
    "confidence_in_assessment": 0.0 to 1.0
}"""


class FactChecker:
    """Validates predictions against expert opinions and analyst consensus."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = LLMClient()

    async def validate(
        self,
        ticker: str,
        company_name: str,
        tech_signals: list[TechnicalSignal],
        intel_items: list[IntelligenceItem],
    ) -> list[ExpertValidation]:
        """Check our analysis against expert opinions."""
        validations = []

        # Extract expert-related intelligence from what we already gathered
        expert_intel = [
            i for i in intel_items
            if i.source_type in (SourceType.EXPERT_OPINION, SourceType.ANALYST_REPORT)
        ]

        # Also search for specific expert mentions
        additional = await self._search_expert_positions(ticker, company_name)
        expert_intel.extend(additional)

        # Check analyst consensus
        consensus = await self._get_analyst_consensus(ticker)
        if consensus:
            validations.append(consensus)

        # Process expert intelligence through LLM for stance detection
        if expert_intel and self.settings.active_api_key:
            stance_results = await self._analyze_expert_stances(ticker, expert_intel)
            validations.extend(stance_results)

        logger.info(
            "fact_check_complete",
            ticker=ticker,
            validations=len(validations),
        )
        return validations

    async def _search_expert_positions(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Search for recent expert positions on the ticker."""
        items = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            # Check whalewisdom for institutional holdings
            try:
                response = await client.get(
                    f"https://whalewisdom.com/stock/{ticker.lower()}",
                    headers=headers,
                )
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)[:3000]
                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.ANALYST_REPORT,
                            source_name="WhaleWisdom",
                            title=f"Institutional holdings for {ticker}",
                            content=text,
                            url=f"https://whalewisdom.com/stock/{ticker.lower()}",
                            credibility_score=0.75,
                        )
                    )
            except Exception as e:
                logger.debug("whalewisdom_error", error=str(e))

            # Check insider trading via OpenInsider
            try:
                response = await client.get(
                    f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&feession=&cession=&sidTicker=&ta=1&tc=1&tp=1&tdlt=&tplt=&trenession=&tession=&gut=&sc=1&vl=&vh=&ocl=&och=&session1=&sic1=&ntfcession=&ntfmession=&nession=",
                    headers=headers,
                )
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    table = soup.find("table", class_="tinytable")
                    if table:
                        text = table.get_text(separator=" ", strip=True)[:2000]
                        items.append(
                            IntelligenceItem(
                                source_type=SourceType.SEC_FILING,
                                source_name="OpenInsider",
                                title=f"Recent insider trades for {ticker}",
                                content=text,
                                url=f"http://openinsider.com/screener?s={ticker}",
                                credibility_score=0.85,
                            )
                        )
            except Exception as e:
                logger.debug("openinsider_error", error=str(e))

        return items

    async def _get_analyst_consensus(self, ticker: str) -> ExpertValidation | None:
        """Get analyst consensus rating from Yahoo Finance."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            rec = stock.recommendations
            if rec is not None and not rec.empty:
                recent = rec.tail(10)
                buy_count = 0
                sell_count = 0
                hold_count = 0
                for _, row in recent.iterrows():
                    grade = str(row.get("To Grade", row.get("toGrade", ""))).lower()
                    if any(w in grade for w in ["buy", "outperform", "overweight"]):
                        buy_count += 1
                    elif any(w in grade for w in ["sell", "underperform", "underweight"]):
                        sell_count += 1
                    else:
                        hold_count += 1

                total = buy_count + sell_count + hold_count
                if total > 0:
                    if buy_count > sell_count:
                        direction = Direction.BULLISH
                    elif sell_count > buy_count:
                        direction = Direction.BEARISH
                    else:
                        direction = Direction.NEUTRAL

                    return ExpertValidation(
                        expert_name="Wall Street Analyst Consensus",
                        agrees_with_prediction=True,
                        expert_direction=direction,
                        credibility_rating=0.7,
                        summary=f"Recent analyst ratings: {buy_count} buy, {hold_count} hold, {sell_count} sell out of {total} ratings.",
                    )
        except Exception as e:
            logger.debug("analyst_consensus_error", error=str(e))
        return None

    async def _analyze_expert_stances(
        self, ticker: str, expert_intel: list[IntelligenceItem]
    ) -> list[ExpertValidation]:
        """Use LLM to extract expert stances from intelligence items."""
        validations = []

        # Group by expert name
        by_expert: dict[str, list[IntelligenceItem]] = {}
        for item in expert_intel:
            name = item.author or item.source_name
            by_expert.setdefault(name, []).append(item)

        # Process each expert
        for expert_name, items in list(by_expert.items())[:8]:
            combined = "\n".join(f"- {i.title}: {i.content[:500]}" for i in items)

            prompt = f"""Expert: {expert_name}
Stock: {ticker}

Recent mentions:
{combined}

What is {expert_name}'s stance on {ticker}?"""

            try:
                result = await self.llm.complete_json(FACT_CHECK_SYSTEM_PROMPT, prompt, max_tokens=256)

                # Find credibility from our tracked list
                credibility = 0.5
                for tracked in TRACKED_EXPERTS:
                    if tracked["name"].lower() in expert_name.lower():
                        credibility = tracked["credibility"]
                        break

                validations.append(
                    ExpertValidation(
                        expert_name=expert_name,
                        source_url=items[0].url if items else "",
                        agrees_with_prediction=result.get("agrees_with_bullish", False),
                        expert_direction=Direction(result.get("expert_direction", "neutral")),
                        credibility_rating=credibility,
                        summary=result.get("summary", ""),
                    )
                )
            except Exception as e:
                logger.debug("expert_analysis_error", expert=expert_name, error=str(e))

        return validations

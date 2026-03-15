"""Aggregates intelligence from all collectors into a unified report."""

from __future__ import annotations

import asyncio

import structlog

from src.core.models import IntelligenceItem
from src.intelligence.news_collector import NewsCollector
from src.intelligence.social_collector import SocialCollector
from src.intelligence.youtube_collector import YouTubeCollector
from src.intelligence.web_scanner import WebScanner

logger = structlog.get_logger()


class IntelligenceAggregator:
    """Orchestrates all intelligence collectors and deduplicates results."""

    def __init__(self):
        self.news = NewsCollector()
        self.social = SocialCollector()
        self.youtube = YouTubeCollector()
        self.web = WebScanner()

    async def gather_all(
        self, ticker: str, company_name: str = ""
    ) -> list[IntelligenceItem]:
        """Run all collectors in parallel and return deduplicated, ranked results."""
        logger.info("gathering_intelligence", ticker=ticker)

        results = await asyncio.gather(
            self.news.collect(ticker, company_name),
            self.social.collect(ticker, company_name),
            self.youtube.collect(ticker, company_name),
            self.web.collect(ticker, company_name),
            return_exceptions=True,
        )

        all_items: list[IntelligenceItem] = []
        source_names = ["news", "social", "youtube", "web"]
        for name, result in zip(source_names, results):
            if isinstance(result, list):
                logger.info("source_collected", source=name, count=len(result))
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("source_failed", source=name, error=str(result))

        # Deduplicate by title similarity
        deduped = self._deduplicate(all_items)

        # Sort by credibility * relevance
        deduped.sort(
            key=lambda x: x.credibility_score * max(x.relevance_score, 0.1),
            reverse=True,
        )

        logger.info(
            "intelligence_gathered",
            ticker=ticker,
            total_raw=len(all_items),
            total_deduped=len(deduped),
        )
        return deduped

    def _deduplicate(self, items: list[IntelligenceItem]) -> list[IntelligenceItem]:
        """Remove near-duplicate items based on title similarity."""
        seen_titles: set[str] = set()
        unique: list[IntelligenceItem] = []

        for item in items:
            normalized = item.title.lower().strip()
            # Simple dedup: skip if we've seen a very similar title
            if normalized in seen_titles:
                continue
            # Check partial overlap
            is_dup = False
            for seen in seen_titles:
                if len(normalized) > 20 and len(seen) > 20:
                    # Check if one title is a substring of another
                    if normalized in seen or seen in normalized:
                        is_dup = True
                        break
            if not is_dup:
                seen_titles.add(normalized)
                unique.append(item)

        return unique

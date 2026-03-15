"""Collect analysis from YouTube financial channels."""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx
import structlog

from src.core.config import get_settings
from src.core.models import IntelligenceItem, SourceType

logger = structlog.get_logger()

# Well-known financial YouTube channels with high credibility
TRUSTED_CHANNELS = {
    "UCL_f53ZEJxp8TtlOkHm5bUg": ("Jordan Belfort", 0.6),
    "UCnMn36GT_H0X-w5_ckLtlgQ": ("Andrei Jikh", 0.7),
    "UCGy7SkBjcIAgTiwkXEtPnYg": ("Aswath Damodaran", 0.9),
    "UCWN3xxRkmTPphYit1FXl3wA": ("Meet Kevin", 0.6),
    "UCbta0n8i6Rljh0obO7HztiA": ("Graham Stephan", 0.65),
    "UCnRCjRmF6kGXPWmBBOGnEew": ("Tom Nash", 0.6),
    "UCfMiRVQJuTj3NpZZP1tKShQ": ("Mark Minervini", 0.8),
    "UCV6KDgJskWaEckne5aPA0aQ": ("Traders4ACause", 0.75),
    "UCIALMKvObZNtJ68-LMxpBDA": ("Peter Lynch", 0.85),
}


class YouTubeCollector:
    """Gathers market analysis from YouTube financial channels."""

    def __init__(self):
        self.settings = get_settings()

    async def collect(self, ticker: str, company_name: str = "") -> list[IntelligenceItem]:
        """Search YouTube for recent stock analysis videos."""
        tasks = [
            self._from_youtube_api(ticker, company_name),
            self._from_youtube_scrape(ticker, company_name),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("youtube_source_failed", error=str(result))
        return items

    async def _from_youtube_api(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Use YouTube Data API to search for analysis videos."""
        if not self.settings.youtube_api_key:
            return []

        items = []
        query = f"{ticker} stock analysis prediction"
        if company_name:
            query = f"{company_name} {ticker} stock analysis"

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                response = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "order": "date",
                        "maxResults": 15,
                        "relevanceLanguage": "en",
                        "publishedAfter": self._recent_cutoff(),
                        "key": self.settings.youtube_api_key,
                    },
                )
                data = response.json()

                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    video_id = item.get("id", {}).get("videoId", "")
                    channel_id = snippet.get("channelId", "")

                    # Boost credibility for known trusted channels
                    channel_name = snippet.get("channelTitle", "")
                    credibility = 0.5
                    if channel_id in TRUSTED_CHANNELS:
                        channel_name, credibility = TRUSTED_CHANNELS[channel_id]

                    published = None
                    if snippet.get("publishedAt"):
                        try:
                            published = datetime.fromisoformat(
                                snippet["publishedAt"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.YOUTUBE,
                            source_name=channel_name,
                            title=snippet.get("title", ""),
                            content=snippet.get("description", "")[:2000],
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            author=channel_name,
                            published_at=published,
                            credibility_score=credibility,
                        )
                    )
            except Exception as e:
                logger.warning("youtube_api_error", error=str(e))
        return items

    async def _from_youtube_scrape(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Fallback: scrape YouTube search results without API key."""
        items = []
        query = f"{ticker} stock analysis prediction today"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            try:
                response = await client.get(
                    "https://www.youtube.com/results",
                    params={"search_query": query, "sp": "CAISBAgBEAE%3D"},  # Today filter
                    headers=headers,
                )
                text = response.text

                # Extract video data from initial page data
                import re

                video_pattern = re.compile(r'"videoId":"([^"]+)"')
                title_pattern = re.compile(r'"title":\{"runs":\[\{"text":"([^"]+)"\}')

                video_ids = video_pattern.findall(text)
                titles = title_pattern.findall(text)

                seen = set()
                for vid, title in zip(video_ids[:10], titles[:10]):
                    if vid in seen:
                        continue
                    seen.add(vid)
                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.YOUTUBE,
                            source_name="YouTube",
                            title=title,
                            content=title,
                            url=f"https://www.youtube.com/watch?v={vid}",
                            credibility_score=0.4,
                        )
                    )
            except Exception as e:
                logger.debug("youtube_scrape_error", error=str(e))
        return items

    def _recent_cutoff(self) -> str:
        """ISO 8601 timestamp for ~3 days ago."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=3)
        return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

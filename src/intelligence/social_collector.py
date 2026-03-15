"""Collect social media intelligence from X/Twitter, Reddit, and StockTwits."""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx
import structlog
from bs4 import BeautifulSoup

from src.core.config import get_settings
from src.core.models import IntelligenceItem, SourceType

logger = structlog.get_logger()


class SocialCollector:
    """Gathers market sentiment from social media platforms."""

    def __init__(self):
        self.settings = get_settings()

    async def collect(self, ticker: str, company_name: str = "") -> list[IntelligenceItem]:
        """Collect social media chatter about a stock."""
        tasks = [
            self._from_twitter(ticker, company_name),
            self._from_reddit(ticker, company_name),
            self._from_stocktwits(ticker),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("social_source_failed", error=str(result))
        return items

    async def _from_twitter(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Fetch tweets about the stock using Twitter API v2."""
        if not self.settings.twitter_bearer_token:
            return []

        items = []
        query = f"${ticker} OR #{ticker} stock -is:retweet lang:en"
        headers = {"Authorization": f"Bearer {self.settings.twitter_bearer_token}"}

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                response = await client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    headers=headers,
                    params={
                        "query": query,
                        "max_results": 50,
                        "tweet.fields": "created_at,author_id,public_metrics",
                    },
                )
                data = response.json()
                for tweet in data.get("data", []):
                    metrics = tweet.get("public_metrics", {})
                    engagement = (
                        metrics.get("like_count", 0)
                        + metrics.get("retweet_count", 0) * 2
                        + metrics.get("reply_count", 0)
                    )
                    # Higher engagement = higher credibility signal
                    cred = min(1.0, engagement / 500)

                    published = None
                    if tweet.get("created_at"):
                        try:
                            published = datetime.fromisoformat(
                                tweet["created_at"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.SOCIAL_MEDIA,
                            source_name="X/Twitter",
                            title=f"Tweet about ${ticker}",
                            content=tweet.get("text", "")[:1000],
                            author=tweet.get("author_id", ""),
                            published_at=published,
                            credibility_score=cred,
                        )
                    )
            except Exception as e:
                logger.warning("twitter_error", error=str(e))
        return items

    async def _from_reddit(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Fetch posts from investing subreddits."""
        items = []
        subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket", "options"]
        headers = {"User-Agent": "StockPredictor/1.0"}

        # Use OAuth if credentials available, otherwise public JSON API
        if self.settings.reddit_client_id and self.settings.reddit_client_secret:
            auth_headers = await self._reddit_oauth()
            if auth_headers:
                headers.update(auth_headers)

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            for sub in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{sub}/search.json"
                    response = await client.get(
                        url,
                        headers=headers,
                        params={
                            "q": f"{ticker} stock",
                            "sort": "new",
                            "limit": 10,
                            "t": "day",
                            "restrict_sr": "true",
                        },
                    )
                    data = response.json()
                    for post in data.get("data", {}).get("children", []):
                        pd = post.get("data", {})
                        score = pd.get("score", 0)
                        cred = min(1.0, score / 200)

                        items.append(
                            IntelligenceItem(
                                source_type=SourceType.REDDIT,
                                source_name=f"r/{sub}",
                                title=pd.get("title", ""),
                                content=(pd.get("selftext", "") or pd.get("title", ""))[:2000],
                                url=f"https://reddit.com{pd.get('permalink', '')}",
                                author=pd.get("author", ""),
                                credibility_score=cred,
                            )
                        )
                except Exception as e:
                    logger.debug("reddit_error", subreddit=sub, error=str(e))
        return items

    async def _reddit_oauth(self) -> dict | None:
        """Get Reddit OAuth token."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(self.settings.reddit_client_id, self.settings.reddit_client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "StockPredictor/1.0"},
                )
                token = response.json().get("access_token")
                if token:
                    return {
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "StockPredictor/1.0",
                    }
        except Exception as e:
            logger.debug("reddit_oauth_error", error=str(e))
        return None

    async def _from_stocktwits(self, ticker: str) -> list[IntelligenceItem]:
        """Fetch messages from StockTwits (no API key needed)."""
        items = []
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            try:
                response = await client.get(
                    f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
                )
                data = response.json()
                for msg in data.get("messages", [])[:20]:
                    sentiment = msg.get("entities", {}).get("sentiment", {})
                    sentiment_label = sentiment.get("basic") if sentiment else None

                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.SOCIAL_MEDIA,
                            source_name="StockTwits",
                            title=f"StockTwits: ${ticker}",
                            content=msg.get("body", "")[:1000],
                            author=msg.get("user", {}).get("username", ""),
                            credibility_score=0.3,  # StockTwits is noisy
                        )
                    )
            except Exception as e:
                logger.debug("stocktwits_error", error=str(e))
        return items

"""Collect news from multiple sources: NewsAPI, RSS feeds, and financial news sites."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import httpx
import feedparser
import structlog
from bs4 import BeautifulSoup

from src.core.config import get_settings
from src.core.models import IntelligenceItem, SourceType

logger = structlog.get_logger()

# Major financial RSS feeds
FINANCIAL_RSS_FEEDS = [
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("Reuters Business", "https://www.reutersagency.com/feed/?best-topics=business-finance"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
]


class NewsCollector:
    """Gathers news from multiple financial sources."""

    def __init__(self):
        self.settings = get_settings()

    async def collect(self, ticker: str, company_name: str = "") -> list[IntelligenceItem]:
        """Collect news about a ticker from all available sources."""
        tasks = [
            self._from_rss_feeds(ticker, company_name),
            self._from_newsapi(ticker, company_name),
            self._from_finviz(ticker),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("news_source_failed", error=str(result))
        return items

    async def _from_rss_feeds(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Parse financial RSS feeds for relevant articles."""
        items = []
        search_terms = {ticker.lower()}
        if company_name:
            search_terms.add(company_name.lower())

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            tasks = []
            for name, url in FINANCIAL_RSS_FEEDS:
                tasks.append(self._fetch_rss(client, name, url, search_terms))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    items.extend(result)
        return items

    async def _fetch_rss(
        self,
        client: httpx.AsyncClient,
        source_name: str,
        url: str,
        search_terms: set[str],
    ) -> list[IntelligenceItem]:
        items = []
        try:
            response = await client.get(url)
            feed = feedparser.parse(response.text)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                combined = f"{title} {summary}".lower()
                if any(term in combined for term in search_terms):
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.RSS_FEED,
                            source_name=source_name,
                            title=title,
                            content=summary[:2000],
                            url=entry.get("link", ""),
                            published_at=published,
                        )
                    )
        except Exception as e:
            logger.debug("rss_feed_error", source=source_name, error=str(e))
        return items

    async def _from_newsapi(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Fetch from NewsAPI if key is available."""
        if not self.settings.news_api_key:
            return []

        items = []
        query = f"{ticker} stock"
        if company_name:
            query = f"{company_name} OR {ticker} stock"

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                response = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                        "language": "en",
                        "apiKey": self.settings.news_api_key,
                    },
                )
                data = response.json()
                for article in data.get("articles", []):
                    published = None
                    if article.get("publishedAt"):
                        try:
                            published = datetime.fromisoformat(
                                article["publishedAt"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    items.append(
                        IntelligenceItem(
                            source_type=SourceType.NEWS,
                            source_name=article.get("source", {}).get("name", "NewsAPI"),
                            title=article.get("title", ""),
                            content=article.get("description", "")[:2000],
                            url=article.get("url", ""),
                            author=article.get("author", ""),
                            published_at=published,
                        )
                    )
            except Exception as e:
                logger.warning("newsapi_error", error=str(e))
        return items

    async def _from_finviz(self, ticker: str) -> list[IntelligenceItem]:
        """Scrape news headlines from Finviz (free tier)."""
        items = []
        url = f"https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=headers)
                soup = BeautifulSoup(response.text, "html.parser")
                news_table = soup.find("table", {"id": "news-table"})
                if not news_table:
                    return items

                rows = news_table.find_all("tr")
                for row in rows[:15]:
                    link_tag = row.find("a")
                    if link_tag:
                        title = link_tag.text.strip()
                        href = link_tag.get("href", "")
                        source_tag = row.find("span", class_="news-link-right")
                        source = source_tag.text.strip() if source_tag else "Finviz"

                        items.append(
                            IntelligenceItem(
                                source_type=SourceType.NEWS,
                                source_name=source,
                                title=title,
                                content=title,
                                url=href,
                            )
                        )
            except Exception as e:
                logger.debug("finviz_error", error=str(e))
        return items

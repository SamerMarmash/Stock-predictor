"""General web scanner for broader market intelligence."""

from __future__ import annotations

import asyncio

import httpx
import structlog
from bs4 import BeautifulSoup

from src.core.config import get_settings
from src.core.models import IntelligenceItem, SourceType

logger = structlog.get_logger()

# Analyst / research sites to scrape
ANALYST_SOURCES = [
    ("TipRanks", "https://www.tipranks.com/stocks/{ticker}/forecast"),
    ("Zacks", "https://www.zacks.com/stock/quote/{ticker}"),
    ("Benzinga", "https://www.benzinga.com/quote/{ticker}"),
    ("TradingView", "https://www.tradingview.com/symbols/{ticker}/"),
]

# Known expert traders / analysts to look for on the web
NOTABLE_TRADERS = [
    "Warren Buffett",
    "Cathie Wood",
    "Ray Dalio",
    "Michael Burry",
    "Stanley Druckenmiller",
    "Bill Ackman",
    "David Tepper",
    "Carl Icahn",
    "George Soros",
    "Jim Cramer",
    "Peter Lynch",
    "Mark Minervini",
    "Paul Tudor Jones",
    "Ken Griffin",
    "Chamath Palihapitiya",
    "Dan Ives",
    "Tom Lee",
]


class WebScanner:
    """Scans the broader web for market intelligence and analyst opinions."""

    def __init__(self):
        self.settings = get_settings()

    async def collect(self, ticker: str, company_name: str = "") -> list[IntelligenceItem]:
        """Scan multiple web sources for analysis and expert opinions."""
        tasks = [
            self._scrape_analyst_sites(ticker),
            self._search_expert_mentions(ticker, company_name),
            self._scrape_sec_filings(ticker),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items = []
        for result in results:
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("web_scanner_failed", error=str(result))
        return items

    async def _scrape_analyst_sites(self, ticker: str) -> list[IntelligenceItem]:
        """Scrape analyst rating sites for consensus views."""
        items = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            for name, url_template in ANALYST_SOURCES:
                url = url_template.format(ticker=ticker)
                try:
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        # Extract text content, focusing on analyst ratings
                        text_content = soup.get_text(separator=" ", strip=True)[:3000]
                        items.append(
                            IntelligenceItem(
                                source_type=SourceType.ANALYST_REPORT,
                                source_name=name,
                                title=f"{name} analysis for {ticker}",
                                content=text_content,
                                url=url,
                                credibility_score=0.7,
                            )
                        )
                except Exception as e:
                    logger.debug("analyst_scrape_error", source=name, error=str(e))
        return items

    async def _search_expert_mentions(
        self, ticker: str, company_name: str
    ) -> list[IntelligenceItem]:
        """Search for mentions of ticker by notable traders/experts."""
        items = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Build search queries for expert mentions
        search_queries = []
        for expert in NOTABLE_TRADERS[:8]:  # Limit to avoid rate limits
            query = f"{expert} {ticker} stock"
            search_queries.append((expert, query))

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            for expert, query in search_queries:
                try:
                    # Use DuckDuckGo HTML search (no API key needed)
                    response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query, "df": "w"},  # Last week
                        headers=headers,
                    )
                    soup = BeautifulSoup(response.text, "html.parser")
                    results = soup.find_all("div", class_="result")

                    for result in results[:3]:
                        title_tag = result.find("a", class_="result__a")
                        snippet_tag = result.find("a", class_="result__snippet")
                        if title_tag:
                            title = title_tag.text.strip()
                            snippet = snippet_tag.text.strip() if snippet_tag else ""
                            href = title_tag.get("href", "")

                            items.append(
                                IntelligenceItem(
                                    source_type=SourceType.EXPERT_OPINION,
                                    source_name=expert,
                                    title=title,
                                    content=snippet[:2000],
                                    url=href,
                                    author=expert,
                                    credibility_score=0.75,
                                )
                            )
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.debug("expert_search_error", expert=expert, error=str(e))
        return items

    async def _scrape_sec_filings(self, ticker: str) -> list[IntelligenceItem]:
        """Check for recent SEC filings (insider trading, 13F, etc.)."""
        items = []
        headers = {
            "User-Agent": "StockPredictor support@example.com",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            try:
                # SEC EDGAR full-text search API
                response = await client.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q": ticker,
                        "dateRange": "custom",
                        "startdt": self._week_ago(),
                        "enddt": self._today(),
                        "forms": "4,8-K,13F-HR,SC 13G,SC 13D",
                    },
                    headers=headers,
                )
                if response.status_code == 200:
                    data = response.json()
                    for filing in data.get("hits", {}).get("hits", [])[:10]:
                        source = filing.get("_source", {})
                        items.append(
                            IntelligenceItem(
                                source_type=SourceType.SEC_FILING,
                                source_name="SEC EDGAR",
                                title=source.get("display_names", [ticker])[0],
                                content=f"Form {source.get('form_type', 'N/A')}: {source.get('entity_name', '')}",
                                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=&dateb=&owner=include&count=10",
                                credibility_score=0.95,
                            )
                        )
            except Exception as e:
                logger.debug("sec_filing_error", error=str(e))
        return items

    def _week_ago(self) -> str:
        from datetime import datetime, timedelta

        return (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    def _today(self) -> str:
        from datetime import datetime

        return datetime.utcnow().strftime("%Y-%m-%d")

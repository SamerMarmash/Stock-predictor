"""Market data fetching via yfinance and optional premium APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


class MarketDataProvider:
    """Fetches real-time and historical market data."""

    def __init__(self):
        self.settings = get_settings()

    async def get_current_price(self, ticker: str) -> dict:
        """Get current price and basic info for a ticker."""
        return await asyncio.to_thread(self._fetch_current, ticker)

    def _fetch_current(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d", interval="1m")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if current_price == 0 and not hist.empty:
            current_price = float(hist["Close"].iloc[-1])

        return {
            "ticker": ticker,
            "company_name": info.get("shortName", ticker),
            "current_price": current_price,
            "previous_close": info.get("previousClose", 0),
            "open": info.get("open", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }

    async def get_historical_data(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        return await asyncio.to_thread(self._fetch_history, ticker, period, interval)

    def _fetch_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            logger.warning("no_historical_data", ticker=ticker, period=period)
        return df

    async def get_intraday_data(self, ticker: str) -> pd.DataFrame:
        """Get today's intraday data (1-minute intervals)."""
        return await self.get_historical_data(ticker, period="1d", interval="1m")

    async def get_options_activity(self, ticker: str) -> dict:
        """Get notable options activity (unusual volume, put/call ratio)."""
        return await asyncio.to_thread(self._fetch_options, ticker)

    def _fetch_options(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        try:
            dates = stock.options
            if not dates:
                return {"available": False}

            nearest = dates[0]
            chain = stock.option_chain(nearest)
            calls_vol = int(chain.calls["volume"].sum()) if "volume" in chain.calls else 0
            puts_vol = int(chain.puts["volume"].sum()) if "volume" in chain.puts else 0
            total = calls_vol + puts_vol
            pc_ratio = puts_vol / calls_vol if calls_vol > 0 else 0

            return {
                "available": True,
                "nearest_expiry": nearest,
                "total_call_volume": calls_vol,
                "total_put_volume": puts_vol,
                "put_call_ratio": round(pc_ratio, 3),
                "total_options_volume": total,
                "signal": "bearish" if pc_ratio > 1.2 else "bullish" if pc_ratio < 0.7 else "neutral",
            }
        except Exception as e:
            logger.warning("options_fetch_failed", ticker=ticker, error=str(e))
            return {"available": False}

    async def get_full_snapshot(self, ticker: str) -> dict:
        """Get a comprehensive snapshot: current data, history, and options."""
        current, history, intraday, options = await asyncio.gather(
            self.get_current_price(ticker),
            self.get_historical_data(ticker),
            self.get_intraday_data(ticker),
            self.get_options_activity(ticker),
        )
        return {
            "current": current,
            "history": history,
            "intraday": intraday,
            "options": options,
        }

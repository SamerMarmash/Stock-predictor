# AI Stock Predictor

AI-powered stock prediction engine that delivers hourly predictions with confidence scoring, powered by multi-source intelligence gathering and professional trading strategies.

## Features

### Multi-Source Intelligence Gathering
- **Financial News**: RSS feeds from Yahoo Finance, MarketWatch, CNBC, Reuters, Bloomberg, Seeking Alpha + NewsAPI + Finviz scraping
- **Social Media**: X/Twitter (API v2), Reddit (r/wallstreetbets, r/stocks, r/investing, r/options), StockTwits
- **YouTube**: YouTube Data API + fallback scraping, with trusted channel tracking (Aswath Damodaran, Mark Minervini, etc.)
- **Web Scanning**: Analyst sites (TipRanks, Zacks, Benzinga, TradingView), SEC EDGAR filings, DuckDuckGo expert search
- **Expert Tracking**: Monitors positions of Warren Buffett, Michael Burry, Cathie Wood, Ray Dalio, and 10+ other notable traders

### Professional Trading Strategies
- **Trend Following** — SMA/EMA crossovers, ADX trend strength, Ichimoku Cloud
- **Mean Reversion** — RSI, Stochastic, Bollinger Bands, Williams %R extremes
- **Breakout Trading** — Range expansion with volume confirmation
- **Scalping Setup** — Fast EMA crossover + RSI + MACD alignment
- **VWAP Strategy** — Institutional level trading with OBV confirmation
- **Divergence Trading** — Price vs RSI/MACD divergence detection
- **Gap Analysis** — Gap-and-go vs gap-fade identification
- **Options Flow** — Put/call ratio and unusual volume analysis

### Technical Indicators (15+)
SMA (20/50/200), EMA (9/21), VWAP, RSI, MACD, Stochastic, Williams %R, Bollinger Bands, ATR, ADX, Ichimoku Cloud, OBV, A/D Line, Volume Spike Detection, Pivot Points

### AI Prediction Engine
- Synthesizes all data through Claude or GPT for final prediction
- Honest confidence scoring (0-100%) with calibrated thresholds
- Fallback to strategy-based consensus when LLM is unavailable
- Bull case and bear case for every prediction

### Fact-Checking System
- Cross-references against Wall Street analyst consensus
- Tracks positions of 15+ notable traders/investors
- Monitors insider trading via OpenInsider
- Checks institutional holdings via WhaleWisdom
- LLM-powered expert stance detection

### Notifications
- Hourly predictions on subscribed stocks
- Multi-channel delivery via Apprise (Discord, Slack, Telegram, Email, Pushover, 80+ services)
- Configurable confidence threshold per subscription
- Rich formatted messages with full prediction details

## Quick Start (Mac)

### Option 1: One-Command Launch (Recommended)

```bash
git clone <repo-url> && cd Stock-predictor
./run.sh
```

That's it. The script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Create a `.env` file from the template
4. Open the interactive dashboard at **http://localhost:8501**

> **No API keys needed** for basic technical analysis. Add an Anthropic or OpenAI key in `.env` for full AI predictions.

### Option 2: Manual Install

```bash
# Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install the package
pip install -e .

# Copy config template
cp .env.example .env

# Launch the dashboard
streamlit run src/ui/app.py
```

### Option 3: CLI Only

```bash
pip install -e .
cp .env.example .env

# One-off prediction
stock-predictor predict AAPL

# Multiple tickers
stock-predictor predict AAPL TSLA NVDA MSFT

# Predict and send notification
stock-predictor predict AAPL --notify
```

### Subscribe to Hourly Predictions

```bash
stock-predictor subscribe AAPL TSLA --threshold 60
stock-predictor start          # Scheduler only
stock-predictor start --api    # Scheduler + REST API
```

### View History

```bash
stock-predictor history AAPL --limit 50
```

## REST API

Start the API server:

```bash
stock-predictor start --api
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/predict/{ticker}` | Generate real-time prediction |
| POST | `/subscribe` | Subscribe to hourly predictions |
| POST | `/unsubscribe` | Unsubscribe from a stock |
| GET | `/subscriptions` | List active subscriptions |
| GET | `/history/{ticker}` | View prediction history |
| GET | `/health` | Health check |

### Example Request

```bash
curl http://localhost:8000/predict/AAPL | python -m json.tool
```

## Architecture

```
src/
├── ui/
│   └── app.py             # Streamlit interactive dashboard
├── core/
│   ├── config.py          # Environment-based configuration
│   ├── models.py          # Domain models (Pydantic)
│   ├── database.py        # Async SQLAlchemy + SQLite
│   ├── market_data.py     # yfinance market data provider
│   └── llm_client.py      # OpenAI/Anthropic unified client
├── intelligence/
│   ├── news_collector.py  # RSS + NewsAPI + Finviz
│   ├── social_collector.py # Twitter + Reddit + StockTwits
│   ├── youtube_collector.py # YouTube API + scraping
│   ├── web_scanner.py     # Analyst sites + SEC + expert search
│   └── aggregator.py      # Dedup + ranking orchestrator
├── strategies/
│   ├── technical_analysis.py # 15+ technical indicators
│   └── trading_strategies.py # 8 professional strategies
├── prediction/
│   ├── engine.py          # Main prediction orchestrator
│   └── fact_checker.py    # Expert validation system
├── notifications/
│   ├── notifier.py        # Multi-channel notifications
│   └── scheduler.py       # APScheduler hourly jobs
├── api/
│   └── routes.py          # FastAPI REST API
└── cli.py                 # Command-line interface
```

## Configuration

All configuration is via environment variables (or `.env` file):

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `TWITTER_BEARER_TOKEN` | No | X/Twitter API for social sentiment |
| `YOUTUBE_API_KEY` | No | YouTube Data API for video analysis |
| `NEWS_API_KEY` | No | NewsAPI for broader news coverage |
| `REDDIT_CLIENT_ID` | No | Reddit API for subreddit analysis |
| `NOTIFICATION_URLS` | No | Comma-separated Apprise URLs |
| `PREDICTION_INTERVAL_MINUTES` | No | Prediction frequency (default: 60) |
| `DEFAULT_CONFIDENCE_THRESHOLD` | No | Min confidence to notify (default: 60) |

*At least one LLM provider key is required.

## Running Tests

```bash
pytest tests/ -v
```

## Disclaimer

This tool is for informational and educational purposes only. It is NOT financial advice. Stock predictions are inherently uncertain. Never invest money you cannot afford to lose based on any prediction system. Always do your own due diligence and consult a licensed financial advisor.

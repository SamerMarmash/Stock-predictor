"""Streamlit UI for AI Stock Predictor."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.config import get_settings
from src.core.market_data import MarketDataProvider
from src.core.models import Direction, StockPrediction, TimeHorizon
from src.strategies.technical_analysis import TechnicalAnalyzer
from src.strategies.trading_strategies import TradingStrategyEngine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4a5568;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prediction banner */
    .prediction-banner {
        border-radius: 16px;
        padding: 28px;
        margin: 16px 0;
        text-align: center;
    }
    .bullish-banner {
        background: linear-gradient(135deg, #1a3a2a 0%, #0d2818 100%);
        border: 2px solid #22c55e;
    }
    .bearish-banner {
        background: linear-gradient(135deg, #3a1a1a 0%, #280d0d 100%);
        border: 2px solid #ef4444;
    }
    .neutral-banner {
        background: linear-gradient(135deg, #2a2a1a 0%, #1e1e0d 100%);
        border: 2px solid #eab308;
    }
    .prediction-direction {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 8px 0;
    }
    .prediction-confidence {
        font-size: 1.3rem;
        opacity: 0.9;
    }
    .prediction-price-target {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 12px 0 4px 0;
    }

    /* Signal pills */
    .signal-bullish {
        display: inline-block;
        background: #166534;
        color: #86efac;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .signal-bearish {
        display: inline-block;
        background: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .signal-neutral {
        display: inline-block;
        background: #3f3f00;
        color: #fde047;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    /* Expert card */
    .expert-card {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
    }

    /* Strategy card */
    .strategy-card {
        background: #1a1f2e;
        border-left: 4px solid;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def direction_color(d: Direction) -> str:
    return {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#eab308"}[d.value]


def direction_emoji(d: Direction) -> str:
    return {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[d.value]


def confidence_color(pct: float) -> str:
    if pct >= 70:
        return "#22c55e"
    if pct >= 50:
        return "#eab308"
    return "#ef4444"


def timeframe_label(horizon: str) -> str:
    """Return a human-readable label for the timeframe."""
    labels = {
        "intraday": "End of Day",
        "daily": "End of Week",
        "weekly": "End of Month",
    }
    return labels.get(horizon, horizon.title())


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def build_candlestick_chart(df: pd.DataFrame, ticker: str, title_suffix: str = "") -> go.Figure:
    """Build an interactive candlestick chart with volume and optional signal annotations."""
    title_text = f"{ticker} {'- ' + title_suffix if title_suffix else 'Price Chart'}"
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "Volume", "RSI"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )

    # Moving averages
    if len(df) >= 20:
        sma20 = df["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color="#3b82f6", width=1.5)),
            row=1, col=1,
        )
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color="#f59e0b", width=1.5)),
            row=1, col=1,
        )

    # Bollinger Bands
    if len(df) >= 20:
        mid = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        fig.add_trace(
            go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.5),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=lower, name="BB Lower", line=dict(color="#6366f1", width=1, dash="dot"), opacity=0.5, fill="tonexty", fillcolor="rgba(99,102,241,0.05)"),
            row=1, col=1,
        )

    # Volume
    if "Volume" in df.columns:
        colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.6),
            row=2, col=1,
        )

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#a78bfa", width=1.5)),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", opacity=0.5, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(text=title_text, font=dict(size=16)),
        height=550,
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#1e2433")
    fig.update_yaxes(gridcolor="#1e2433")

    return fig


def build_prediction_chart(
    history: pd.DataFrame,
    ticker: str,
    current_price: float,
    predicted_price: float,
    time_horizon: str,
    direction: Direction,
) -> go.Figure:
    """Build a projected price chart showing the prediction path."""
    # Determine how many future points to show based on timeframe
    horizon_days = {"intraday": 1, "daily": 5, "weekly": 22}
    n_future = horizon_days.get(time_horizon, 5)

    # Get last N days of actual data for context
    recent = history.tail(30) if len(history) > 30 else history

    # Generate projected price path
    last_date = recent.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=n_future + 1)[1:]

    # Create a smooth curve from current to predicted price
    steps = len(future_dates)
    if steps > 0:
        # Add some realistic variance to the projected path
        t = np.linspace(0, 1, steps)
        # Use a slight S-curve for more natural look
        smooth = 3 * t**2 - 2 * t**3
        projected_prices = current_price + (predicted_price - current_price) * smooth
        # Add small random noise for realism
        noise = np.random.normal(0, abs(predicted_price - current_price) * 0.03, steps)
        noise[-1] = 0  # Ensure final point hits target exactly
        projected_prices = projected_prices + noise
        projected_prices[-1] = predicted_price
    else:
        future_dates = [last_date + pd.Timedelta(days=1)]
        projected_prices = [predicted_price]

    color = direction_color(direction)

    fig = go.Figure()

    # Recent actual price line
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent["Close"],
        name="Recent Price",
        line=dict(color="#a0aec0", width=2),
        mode="lines",
    ))

    # Projection line
    # Connect from last actual price to projected
    all_proj_dates = [recent.index[-1]] + list(future_dates)
    all_proj_prices = [current_price] + list(projected_prices)

    fig.add_trace(go.Scatter(
        x=all_proj_dates,
        y=all_proj_prices,
        name="Predicted Path",
        line=dict(color=color, width=3, dash="dot"),
        mode="lines",
    ))

    # Confidence band (wider as we go further)
    spread = abs(predicted_price - current_price)
    upper_band = [current_price] + [p + spread * 0.3 * (i + 1) / steps for i, p in enumerate(projected_prices)]
    lower_band = [current_price] + [p - spread * 0.3 * (i + 1) / steps for i, p in enumerate(projected_prices)]

    fig.add_trace(go.Scatter(
        x=all_proj_dates,
        y=upper_band,
        name="Upper Range",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=all_proj_dates,
        y=lower_band,
        name="Confidence Range",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},0.15)",
    ))

    # Target price marker
    fig.add_trace(go.Scatter(
        x=[future_dates[-1]],
        y=[predicted_price],
        name=f"Target: ${predicted_price:.2f}",
        mode="markers+text",
        marker=dict(color=color, size=14, symbol="star"),
        text=[f"${predicted_price:.2f}"],
        textposition="top center",
        textfont=dict(color=color, size=14, family="Arial Black"),
    ))

    # Divider line between actual and projected
    divider_x = recent.index[-1].isoformat()
    fig.add_shape(
        type="line",
        x0=divider_x, x1=divider_x,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="#4a5568", width=1, dash="dash"),
        opacity=0.7,
    )
    fig.add_annotation(
        x=divider_x, y=1.05, yref="paper",
        text="Now", showarrow=False,
        font=dict(color="#a0aec0", size=12),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(text=f"{ticker} - Predicted ({timeframe_label(time_horizon)})", font=dict(size=16)),
        height=550,
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#1e2433")
    fig.update_yaxes(gridcolor="#1e2433")

    return fig


def build_gauge_chart(confidence: float, title: str = "Confidence") -> go.Figure:
    """Build a confidence gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568"},
            "bar": {"color": confidence_color(confidence)},
            "bgcolor": "#1a1f2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(239,68,68,0.15)"},
                {"range": [35, 65], "color": "rgba(234,179,8,0.15)"},
                {"range": [65, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": confidence,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        title=dict(text=title, font=dict(size=14, color="#a0aec0")),
    )
    return fig


def build_signal_breakdown_chart(signals) -> go.Figure:
    """Horizontal bar chart showing signal strengths."""
    names = [s.indicator_name for s in signals]
    strengths = [s.strength for s in signals]
    colors = [direction_color(s.signal) for s in signals]

    fig = go.Figure(go.Bar(
        x=strengths,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0%}" for s in strengths],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(250, len(names) * 32),
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Technical Signals", font=dict(size=14, color="#a0aec0")),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
    )
    return fig


def build_strategy_radar(strategies) -> go.Figure:
    """Radar chart of strategy confidence levels."""
    names = [s.strategy_name for s in strategies]
    values = [s.confidence for s in strategies]
    colors = [direction_color(s.direction) for s in strategies]

    # Close the radar
    names_r = names + [names[0]]
    values_r = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_r,
        theta=names_r,
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color=colors + [colors[0]], size=8),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(text="Strategy Analysis", font=dict(size=14, color="#a0aec0")),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#2d3348"),
            angularaxis=dict(gridcolor="#2d3348"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar - Simplified: just ticker and timeframe
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 AI Stock Predictor")
    st.markdown("---")

    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        max_chars=10,
        help="Enter a stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
    ).upper().strip()

    st.markdown("### Prediction Timeframe")

    time_horizon = st.selectbox(
        "How far out?",
        ["Intraday", "Daily", "Weekly"],
        index=1,
        help="Intraday = end of today, Daily = end of week, Weekly = end of month",
    )

    st.markdown("---")

    settings = get_settings()
    has_llm = bool(settings.active_api_key)

    if has_llm:
        st.success(f"LLM: {settings.llm_provider} ({settings.llm_model})")
    else:
        st.warning("No LLM API key set. Using technical-only mode.")

    analyze_btn = st.button(
        "🔍 Analyze Stock",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption("All intelligence sources are searched automatically.")
    st.caption("Predictions are for educational purposes only. Not financial advice.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# 📈 AI Stock Predictor")

if not ticker:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

# Always show the chart on page load
market = MarketDataProvider()

# Fetch basic data
with st.spinner(f"Loading {ticker} market data..."):
    try:
        current_data = run_async(market.get_current_price(ticker))
        history = run_async(market.get_historical_data(ticker, period="3mo"))
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        st.stop()

if not current_data:
    st.error(f"Could not find stock data for '{ticker}'. Check the ticker symbol.")
    st.stop()

# Company header
company = current_data.get("company_name", ticker)
price = current_data.get("current_price", 0)
prev_close = current_data.get("previous_close", price)
change = price - prev_close
change_pct = (change / prev_close * 100) if prev_close else 0

col_title, col_price = st.columns([3, 1])
with col_title:
    st.markdown(f"### {company} ({ticker})")
with col_price:
    color = "#22c55e" if change >= 0 else "#ef4444"
    arrow = "▲" if change >= 0 else "▼"
    st.markdown(
        f'<div style="text-align:right">'
        f'<span style="font-size:1.8rem;font-weight:700">${price:.2f}</span><br>'
        f'<span style="color:{color};font-size:1.1rem">{arrow} {change:+.2f} ({change_pct:+.2f}%)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Metric cards row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Open</div><div class="metric-value">${current_data.get("open", 0):.2f}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Day High</div><div class="metric-value">${current_data.get("day_high", 0):.2f}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Day Low</div><div class="metric-value">${current_data.get("day_low", 0):.2f}</div></div>', unsafe_allow_html=True)
with m4:
    vol = current_data.get("volume", 0)
    vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K" if vol >= 1e3 else str(vol)
    st.markdown(f'<div class="metric-card"><div class="metric-label">Volume</div><div class="metric-value">{vol_str}</div></div>', unsafe_allow_html=True)
with m5:
    mcap = current_data.get("market_cap", 0)
    if mcap >= 1e12:
        mcap_str = f"${mcap/1e12:.2f}T"
    elif mcap >= 1e9:
        mcap_str = f"${mcap/1e9:.1f}B"
    elif mcap >= 1e6:
        mcap_str = f"${mcap/1e6:.0f}M"
    else:
        mcap_str = f"${mcap:,.0f}"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{mcap_str}</div></div>', unsafe_allow_html=True)

# Chart - shown before analysis
if history is not None and not history.empty:
    chart = build_candlestick_chart(history, ticker, title_suffix="Current")
    st.plotly_chart(chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis (on button press)
# ---------------------------------------------------------------------------
if analyze_btn:
    st.markdown("---")
    st.markdown("## 🔬 Deep Analysis Results")

    # Technical Analysis
    tech_analyzer = TechnicalAnalyzer()
    strategy_engine = TradingStrategyEngine()

    with st.spinner("Running technical analysis..."):
        tech_signals = tech_analyzer.analyze(history) if history is not None and not history.empty else []
        strategy_results = strategy_engine.evaluate(
            tech_signals, current_data, history
        ) if tech_signals else []

    # Intelligence Gathering - ALWAYS use ALL sources
    intel_items = []
    with st.spinner("Searching all intelligence sources (news, social, YouTube, web, insider data)..."):
        try:
            from src.intelligence.aggregator import IntelligenceAggregator
            aggregator = IntelligenceAggregator()

            async def gather_intel():
                tasks = [
                    aggregator.news.collect(ticker, company),
                    aggregator.social.collect(ticker, company),
                    aggregator.youtube.collect(ticker, company),
                    aggregator.web.collect(ticker, company),
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                items = []
                for r in results:
                    if isinstance(r, list):
                        items.extend(r)
                return items

            intel_items = run_async(gather_intel())
        except Exception as e:
            st.warning(f"Intelligence gathering partial failure: {e}")

    # Fact checking
    expert_validations = []
    if intel_items and has_llm:
        with st.spinner("Fact-checking against expert opinions & insider activity..."):
            try:
                from src.prediction.fact_checker import FactChecker
                checker = FactChecker()
                expert_validations = run_async(
                    checker.validate(ticker, company, tech_signals, intel_items)
                )
            except Exception as e:
                st.warning(f"Fact-checking error: {e}")

    # AI Prediction with deep thinking
    prediction = None
    if has_llm:
        with st.spinner("AI is deeply analyzing all data with extended thinking (this may take a moment)..."):
            try:
                from src.prediction.engine import PredictionEngine
                engine = PredictionEngine()
                prediction = run_async(engine.predict(ticker, time_horizon=time_horizon.lower()))
            except Exception as e:
                st.warning(f"AI prediction error: {e}")

    # If no LLM, build a technical-only prediction
    if prediction is None and strategy_results:
        bullish = sum(1 for s in strategy_results if s.direction == Direction.BULLISH)
        bearish = sum(1 for s in strategy_results if s.direction == Direction.BEARISH)
        total = len(strategy_results)

        if bullish > bearish:
            direction = Direction.BULLISH
            raw_conf = bullish / total
        elif bearish > bullish:
            direction = Direction.BEARISH
            raw_conf = bearish / total
        else:
            direction = Direction.NEUTRAL
            raw_conf = 0.4

        avg_confidence = sum(s.confidence for s in strategy_results) / total if total else 0.4
        final_conf = round(min(75, avg_confidence * 100 * raw_conf + 10), 1)
        pred_change = round((final_conf / 100 - 0.5) * 3, 2)
        if direction == Direction.BEARISH:
            pred_change = -abs(pred_change)

        prediction = StockPrediction(
            ticker=ticker,
            company_name=company,
            direction=direction,
            predicted_price_change_pct=pred_change,
            confidence_pct=final_conf,
            current_price=price,
            predicted_price=round(price * (1 + pred_change / 100), 2),
            time_horizon=TimeHorizon(time_horizon.lower()),
            key_drivers=[s.reasoning for s in strategy_results[:3]],
            intelligence_summary="Technical analysis only (no LLM API key configured).",
            strategy_signals=strategy_results,
            expert_validations=expert_validations,
            source_count=len(intel_items),
            bull_case="Technical indicators lean bullish." if direction == Direction.BULLISH else "Some support signals detected.",
            bear_case="Technical indicators lean bearish." if direction == Direction.BEARISH else "Some resistance signals detected.",
            model_used="Technical-only (no LLM)",
        )

    # -----------------------------------------------------------------------
    # Display Prediction
    # -----------------------------------------------------------------------
    if prediction:
        d = prediction.direction
        banner_class = f"{d.value}-banner"
        emoji = direction_emoji(d)
        horizon_text = timeframe_label(prediction.time_horizon.value)

        # Main prediction banner with predicted price prominently displayed
        st.markdown(
            f'<div class="prediction-banner {banner_class}">'
            f'<div class="prediction-direction" style="color:{direction_color(d)}">'
            f'{emoji} {d.value.upper()}</div>'
            f'<div class="prediction-price-target" style="color:white">'
            f'Predicted Price ({horizon_text}): '
            f'<span style="color:{direction_color(d)}">${prediction.predicted_price:.2f}</span>'
            f'</div>'
            f'<div style="font-size:1.1rem;color:#a0aec0;margin:4px 0">'
            f'Current: ${prediction.current_price:.2f} → Target: ${prediction.predicted_price:.2f} '
            f'({prediction.predicted_price_change_pct:+.1f}%)</div>'
            f'<div class="prediction-confidence" style="margin-top:8px">'
            f'AI Confidence: <strong style="color:{confidence_color(prediction.confidence_pct)}">'
            f'{prediction.confidence_pct:.0f}%</strong></div>'
            f'<div style="margin-top:6px;color:#718096;font-size:0.85rem">'
            f'Sources Analyzed: {prediction.source_count} | '
            f'Model: {prediction.model_used}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Side-by-side charts: Current vs Predicted
        st.markdown("### Current Price vs Prediction")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            if history is not None and not history.empty:
                current_chart = build_candlestick_chart(history, ticker, title_suffix="Current")
                st.plotly_chart(current_chart, use_container_width=True)
        with chart_right:
            if history is not None and not history.empty:
                pred_chart = build_prediction_chart(
                    history, ticker,
                    prediction.current_price,
                    prediction.predicted_price,
                    prediction.time_horizon.value,
                    prediction.direction,
                )
                st.plotly_chart(pred_chart, use_container_width=True)

        # Confidence gauge + strategy radar
        gcol, rcol = st.columns(2)
        with gcol:
            st.plotly_chart(build_gauge_chart(prediction.confidence_pct), use_container_width=True)
        with rcol:
            if strategy_results:
                st.plotly_chart(build_strategy_radar(strategy_results), use_container_width=True)

        # Tabs for detailed analysis
        tab_strat, tab_tech, tab_intel, tab_experts, tab_cases = st.tabs([
            "📊 Strategies", "📉 Technical Signals", "🌐 Intelligence", "👤 Expert Validation", "📋 Upside / Downside",
        ])

        with tab_strat:
            if strategy_results:
                for s in strategy_results:
                    color = direction_color(s.direction)
                    st.markdown(
                        f'<div class="strategy-card" style="border-left-color:{color}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<strong>{s.strategy_name}</strong>'
                        f'<span class="signal-{s.direction.value}">{s.direction.value.upper()} ({s.confidence:.0%})</span>'
                        f'</div>'
                        f'<div style="color:#a0aec0;margin-top:6px;font-size:0.9rem">{s.reasoning}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No strategy signals available.")

        with tab_tech:
            if tech_signals:
                st.plotly_chart(build_signal_breakdown_chart(tech_signals), use_container_width=True)
                with st.expander("Signal Details", expanded=False):
                    for s in tech_signals:
                        pill_class = f"signal-{s.signal.value}"
                        st.markdown(
                            f'<span class="{pill_class}">{s.indicator_name}: {s.signal.value.upper()}</span> '
                            f'<span style="color:#a0aec0;font-size:0.85rem">{s.description}</span>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No technical signals available.")

        with tab_intel:
            if intel_items:
                st.markdown(f"**{len(intel_items)} intelligence items gathered**")

                # Group by source type
                from collections import Counter
                type_counts = Counter(i.source_type.value for i in intel_items)
                tcols = st.columns(min(len(type_counts), 4))
                for idx, (stype, count) in enumerate(type_counts.most_common()):
                    with tcols[idx % len(tcols)]:
                        st.metric(stype.replace("_", " ").title(), count)

                with st.expander("All Sources", expanded=False):
                    for item in intel_items[:50]:
                        cred_pct = f"{item.credibility_score:.0%}"
                        st.markdown(
                            f"**[{item.source_name}]** {item.title[:100]} "
                            f"*(credibility: {cred_pct})*"
                        )
                        if item.url:
                            st.caption(item.url)
            else:
                st.info("No intelligence gathered.")

        with tab_experts:
            if prediction.expert_validations:
                for ev in prediction.expert_validations:
                    agree_icon = "✅" if ev.agrees_with_prediction else "❌"
                    dir_color = direction_color(ev.expert_direction)
                    cred_stars = "⭐" * int(ev.credibility_rating * 5)
                    st.markdown(
                        f'<div class="expert-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<strong>{ev.expert_name}</strong>'
                        f'<span>{cred_stars} ({ev.credibility_rating:.0%})</span>'
                        f'</div>'
                        f'<div style="margin-top:6px">'
                        f'{agree_icon} <span style="color:{dir_color}">{ev.expert_direction.value.upper()}</span> '
                        f'<span style="color:#a0aec0"> — {ev.summary}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No expert validations available. Add an LLM API key for fact-checking.")

        with tab_cases:
            bc, brc = st.columns(2)
            with bc:
                st.markdown(
                    f'<div class="strategy-card" style="border-left-color:#22c55e">'
                    f'<strong style="color:#22c55e">📈 Upside Potential</strong>'
                    f'<div style="margin-top:8px;color:#d1d5db">{prediction.bull_case}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with brc:
                st.markdown(
                    f'<div class="strategy-card" style="border-left-color:#ef4444">'
                    f'<strong style="color:#ef4444">📉 Downside Risk</strong>'
                    f'<div style="margin-top:8px;color:#d1d5db">{prediction.bear_case}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if prediction.key_drivers:
                st.markdown("#### Key Drivers")
                for i, driver in enumerate(prediction.key_drivers, 1):
                    st.markdown(f"{i}. {driver}")

    elif not tech_signals:
        st.warning("Could not generate analysis. Check the ticker symbol and try again.")

elif not analyze_btn:
    # Show quick stats before analysis
    st.markdown("---")
    st.info("👆 Click **Analyze Stock** in the sidebar to run the full AI prediction engine.")

    # Show basic info
    if current_data:
        info_cols = st.columns(3)
        with info_cols[0]:
            pe = current_data.get("pe_ratio")
            st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        with info_cols[1]:
            st.metric("Sector", current_data.get("sector", "N/A"))
        with info_cols[2]:
            wk52 = current_data.get("52_week_range", "N/A")
            st.metric("52-Week Range", wk52)

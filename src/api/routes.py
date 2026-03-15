"""FastAPI REST API for the stock prediction service."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from src.core.database import PredictionRecord, SubscriptionRecord, get_session
from src.notifications.notifier import Notifier
from src.notifications.scheduler import PredictionScheduler
from src.prediction.engine import PredictionEngine

app = FastAPI(
    title="AI Stock Predictor",
    description="AI-powered stock predictions with multi-source intelligence and professional trading strategies",
    version="0.1.0",
)

engine = PredictionEngine()
scheduler = PredictionScheduler()
notifier = Notifier()


class SubscribeRequest(BaseModel):
    ticker: str
    confidence_threshold: int = 60
    user_id: str = "default"


class UnsubscribeRequest(BaseModel):
    ticker: str
    user_id: str = "default"


@app.on_event("startup")
async def startup():
    pass  # Scheduler started via CLI flag


@app.get("/")
async def root():
    return {
        "service": "AI Stock Predictor",
        "version": "0.1.0",
        "endpoints": [
            "/predict/{ticker}",
            "/subscribe",
            "/unsubscribe",
            "/subscriptions",
            "/history/{ticker}",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/predict/{ticker}")
async def predict(ticker: str):
    """Generate a real-time prediction for a stock ticker."""
    try:
        prediction = await engine.predict(ticker.upper())
        return prediction.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subscribe")
async def subscribe(req: SubscribeRequest):
    """Subscribe to hourly predictions for a stock."""
    await scheduler.add_subscription(
        req.ticker, req.confidence_threshold, req.user_id
    )
    return {
        "status": "subscribed",
        "ticker": req.ticker.upper(),
        "confidence_threshold": req.confidence_threshold,
    }


@app.post("/unsubscribe")
async def unsubscribe(req: UnsubscribeRequest):
    """Unsubscribe from a stock's predictions."""
    await scheduler.remove_subscription(req.ticker, req.user_id)
    return {"status": "unsubscribed", "ticker": req.ticker.upper()}


@app.get("/subscriptions")
async def list_subscriptions(user_id: str = "default"):
    """List all current subscriptions."""
    session = await get_session()
    async with session:
        result = await session.execute(
            select(SubscriptionRecord).where(SubscriptionRecord.user_id == user_id)
        )
        records = result.scalars().all()
        return [
            {
                "ticker": r.ticker,
                "confidence_threshold": r.confidence_threshold,
                "notify": r.notify,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]


@app.get("/history/{ticker}")
async def prediction_history(
    ticker: str,
    limit: int = Query(default=20, le=100),
):
    """Get prediction history for a ticker."""
    session = await get_session()
    async with session:
        result = await session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.ticker == ticker.upper())
            .order_by(PredictionRecord.timestamp.desc())
            .limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "direction": r.direction,
                "predicted_change_pct": r.predicted_change_pct,
                "confidence_pct": r.confidence_pct,
                "current_price": r.current_price,
                "predicted_price": r.predicted_price,
                "time_horizon": r.time_horizon,
                "key_drivers": json.loads(r.key_drivers) if r.key_drivers else [],
                "intelligence_summary": r.intelligence_summary,
                "source_count": r.source_count,
                "model_used": r.model_used,
                "actual_price": r.actual_price,
                "was_correct": r.was_correct,
            }
            for r in records
        ]

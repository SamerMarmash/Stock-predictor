"""Scheduler for recurring hourly predictions on subscribed stocks."""

from __future__ import annotations

import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
import structlog

from src.core.config import get_settings
from src.core.database import PredictionRecord, SubscriptionRecord, get_session
from src.core.models import Subscription
from src.notifications.notifier import Notifier
from src.prediction.engine import PredictionEngine

logger = structlog.get_logger()


class PredictionScheduler:
    """Manages hourly prediction jobs for subscribed stocks."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = PredictionEngine()
        self.notifier = Notifier()
        self.scheduler = AsyncIOScheduler()

    async def start(self):
        """Start the scheduler."""
        self.scheduler.add_job(
            self._run_predictions,
            IntervalTrigger(minutes=self.settings.prediction_interval_minutes),
            id="prediction_cycle",
            name="Hourly stock predictions",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(
            "scheduler_started",
            interval_minutes=self.settings.prediction_interval_minutes,
        )
        # Run immediately on start
        await self._run_predictions()

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")

    async def _run_predictions(self):
        """Run predictions for all subscribed tickers."""
        subscriptions = await self._get_subscriptions()
        if not subscriptions:
            logger.info("no_subscriptions")
            return

        logger.info("prediction_cycle_start", count=len(subscriptions))

        for sub in subscriptions:
            try:
                prediction = await self.engine.predict(sub.ticker)

                # Save to database
                await self._save_prediction(prediction)

                # Send notification if confidence meets threshold
                if sub.notify and prediction.confidence_pct >= sub.confidence_threshold:
                    await self.notifier.send(prediction)
                else:
                    logger.info(
                        "notification_skipped",
                        ticker=sub.ticker,
                        confidence=prediction.confidence_pct,
                        threshold=sub.confidence_threshold,
                    )
            except Exception as e:
                logger.error("prediction_failed", ticker=sub.ticker, error=str(e))

    async def _get_subscriptions(self) -> list[Subscription]:
        """Load subscriptions from database."""
        session = await get_session()
        async with session:
            result = await session.execute(
                select(SubscriptionRecord).where(SubscriptionRecord.notify == True)
            )
            records = result.scalars().all()
            return [
                Subscription(
                    ticker=r.ticker,
                    user_id=r.user_id,
                    confidence_threshold=r.confidence_threshold,
                    notify=r.notify,
                )
                for r in records
            ]

    async def _save_prediction(self, prediction):
        """Save a prediction to the database."""
        session = await get_session()
        async with session:
            record = PredictionRecord(
                ticker=prediction.ticker,
                timestamp=prediction.timestamp,
                direction=prediction.direction.value,
                predicted_change_pct=prediction.predicted_price_change_pct,
                confidence_pct=prediction.confidence_pct,
                current_price=prediction.current_price,
                predicted_price=prediction.predicted_price,
                time_horizon=prediction.time_horizon.value,
                intelligence_summary=prediction.intelligence_summary,
                key_drivers=json.dumps(prediction.key_drivers),
                source_count=prediction.source_count,
                model_used=prediction.model_used,
            )
            session.add(record)
            await session.commit()

    async def add_subscription(
        self, ticker: str, confidence_threshold: int = 60, user_id: str = "default"
    ):
        """Add a new stock subscription."""
        session = await get_session()
        async with session:
            # Check if already subscribed
            result = await session.execute(
                select(SubscriptionRecord).where(
                    SubscriptionRecord.ticker == ticker.upper(),
                    SubscriptionRecord.user_id == user_id,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.confidence_threshold = confidence_threshold
                existing.notify = True
            else:
                session.add(
                    SubscriptionRecord(
                        ticker=ticker.upper(),
                        user_id=user_id,
                        confidence_threshold=confidence_threshold,
                        notify=True,
                    )
                )
            await session.commit()
        logger.info("subscription_added", ticker=ticker.upper(), threshold=confidence_threshold)

    async def remove_subscription(self, ticker: str, user_id: str = "default"):
        """Remove a stock subscription."""
        session = await get_session()
        async with session:
            result = await session.execute(
                select(SubscriptionRecord).where(
                    SubscriptionRecord.ticker == ticker.upper(),
                    SubscriptionRecord.user_id == user_id,
                )
            )
            record = result.scalar_one_or_none()
            if record:
                await session.delete(record)
                await session.commit()
                logger.info("subscription_removed", ticker=ticker.upper())

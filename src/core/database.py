"""Async database setup using SQLAlchemy + aiosqlite."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.core.config import get_settings


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    direction = Column(String(10), nullable=False)
    predicted_change_pct = Column(Float, nullable=False)
    confidence_pct = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    time_horizon = Column(String(20), nullable=False)
    intelligence_summary = Column(Text, default="")
    key_drivers = Column(Text, default="")  # JSON-encoded list
    source_count = Column(Integer, default=0)
    model_used = Column(String(100), default="")
    actual_price = Column(Float, nullable=True)  # Filled in later for accuracy tracking
    was_correct = Column(Boolean, nullable=True)


class SubscriptionRecord(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    user_id = Column(String(100), default="default")
    confidence_threshold = Column(Integer, default=60)
    notify = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = None
_session_factory = None


async def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False)
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    return _engine


async def get_session() -> AsyncSession:
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory()

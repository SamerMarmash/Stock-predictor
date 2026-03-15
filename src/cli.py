"""Command-line interface for the stock prediction app."""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog


def setup_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


async def cmd_predict(args):
    """Run a one-off prediction for a ticker."""
    from src.prediction.engine import PredictionEngine
    from src.notifications.notifier import Notifier

    engine = PredictionEngine()
    notifier = Notifier()

    for ticker in args.tickers:
        print(f"\n{'='*60}")
        print(f"  Generating prediction for {ticker.upper()}...")
        print(f"{'='*60}\n")

        prediction = await engine.predict(ticker)
        print(notifier.format_console(prediction))

        if args.notify:
            await notifier.send(prediction)


async def cmd_subscribe(args):
    """Subscribe to a stock for hourly predictions."""
    from src.notifications.scheduler import PredictionScheduler

    scheduler = PredictionScheduler()
    for ticker in args.tickers:
        await scheduler.add_subscription(
            ticker, args.threshold, args.user_id
        )
        print(f"Subscribed to {ticker.upper()} (threshold: {args.threshold}%)")


async def cmd_unsubscribe(args):
    """Unsubscribe from a stock."""
    from src.notifications.scheduler import PredictionScheduler

    scheduler = PredictionScheduler()
    for ticker in args.tickers:
        await scheduler.remove_subscription(ticker, args.user_id)
        print(f"Unsubscribed from {ticker.upper()}")


async def cmd_start(args):
    """Start the prediction service (scheduler + optional API)."""
    from src.notifications.scheduler import PredictionScheduler

    scheduler = PredictionScheduler()

    if args.api:
        import uvicorn
        from src.api.routes import app

        # Start scheduler in background
        asyncio.create_task(scheduler.start())

        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    else:
        await scheduler.start()
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, asyncio.CancelledError):
            scheduler.stop()
            print("\nScheduler stopped.")


async def cmd_history(args):
    """Show prediction history for a ticker."""
    import json

    from sqlalchemy import select

    from src.core.database import PredictionRecord, get_session

    session = await get_session()
    async with session:
        result = await session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.ticker == args.ticker.upper())
            .order_by(PredictionRecord.timestamp.desc())
            .limit(args.limit)
        )
        records = result.scalars().all()

    if not records:
        print(f"No prediction history for {args.ticker.upper()}")
        return

    for r in records:
        direction_icon = "🟢" if r.direction == "bullish" else "🔴" if r.direction == "bearish" else "⚪"
        print(
            f"{direction_icon} [{r.timestamp}] {r.ticker}: {r.direction} "
            f"({r.confidence_pct:.0f}% conf) "
            f"${r.current_price:.2f} → ${r.predicted_price:.2f} "
            f"({r.predicted_change_pct:+.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="AI Stock Predictor - Professional-grade stock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stock-predictor predict AAPL TSLA NVDA     One-off prediction
  stock-predictor predict AAPL --notify      Predict and send notification
  stock-predictor subscribe AAPL MSFT        Subscribe to hourly predictions
  stock-predictor start                      Start the scheduler
  stock-predictor start --api                Start with REST API server
  stock-predictor history AAPL               View prediction history
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_predict = subparsers.add_parser("predict", help="Generate a prediction")
    p_predict.add_argument("tickers", nargs="+", help="Stock ticker(s)")
    p_predict.add_argument("--notify", action="store_true", help="Send notification")
    p_predict.set_defaults(func=cmd_predict)

    # subscribe
    p_sub = subparsers.add_parser("subscribe", help="Subscribe to hourly predictions")
    p_sub.add_argument("tickers", nargs="+", help="Stock ticker(s)")
    p_sub.add_argument("--threshold", type=int, default=60, help="Min confidence to notify (%%)")
    p_sub.add_argument("--user-id", default="default", help="User ID")
    p_sub.set_defaults(func=cmd_subscribe)

    # unsubscribe
    p_unsub = subparsers.add_parser("unsubscribe", help="Unsubscribe from predictions")
    p_unsub.add_argument("tickers", nargs="+", help="Stock ticker(s)")
    p_unsub.add_argument("--user-id", default="default", help="User ID")
    p_unsub.set_defaults(func=cmd_unsubscribe)

    # start
    p_start = subparsers.add_parser("start", help="Start the prediction service")
    p_start.add_argument("--api", action="store_true", help="Also start REST API")
    p_start.add_argument("--host", default="0.0.0.0", help="API host")
    p_start.add_argument("--port", type=int, default=8000, help="API port")
    p_start.set_defaults(func=cmd_start)

    # history
    p_hist = subparsers.add_parser("history", help="View prediction history")
    p_hist.add_argument("ticker", help="Stock ticker")
    p_hist.add_argument("--limit", type=int, default=20, help="Max records")
    p_hist.set_defaults(func=cmd_history)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()

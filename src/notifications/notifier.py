"""Notification system using Apprise for multi-channel delivery.

Supports: Discord, Slack, Telegram, Email, Pushover, and 80+ other services.
"""

from __future__ import annotations

import json

import structlog
from jinja2 import Template

from src.core.config import get_settings
from src.core.models import StockPrediction

logger = structlog.get_logger()

NOTIFICATION_TEMPLATE = Template("""📊 Stock Prediction: {{ prediction.ticker }} ({{ prediction.company_name }})
━━━━━━━━━━━━━━━━━━━━━━━

{% if prediction.direction.value == "bullish" %}🟢 BULLISH{% elif prediction.direction.value == "bearish" %}🔴 BEARISH{% else %}⚪ NEUTRAL{% endif %} | Confidence: {{ "%.1f"|format(prediction.confidence_pct) }}%

💰 Current: ${{ "%.2f"|format(prediction.current_price) }} → Target: ${{ "%.2f"|format(prediction.predicted_price) }} ({{ "%+.1f"|format(prediction.predicted_price_change_pct) }}%)
⏱️ Horizon: {{ prediction.time_horizon.value }}

🔑 Key Drivers:
{% for driver in prediction.key_drivers[:5] %}- {{ driver }}
{% endfor %}

📈 Strategy Signals:
{% for s in prediction.strategy_signals[:5] %}- {{ s.strategy_name }}: {{ s.direction.value }} ({{ "%.0f"|format(s.confidence * 100) }}%)
{% endfor %}

{% if prediction.expert_validations %}👥 Expert Validations:
{% for v in prediction.expert_validations[:3] %}- {{ v.expert_name }}: {{ v.expert_direction.value }} (credibility: {{ "%.0f"|format(v.credibility_rating * 100) }}%)
{% endfor %}{% endif %}

📋 Summary: {{ prediction.intelligence_summary }}

🐂 Bull case: {{ prediction.bull_case }}
🐻 Bear case: {{ prediction.bear_case }}

📡 Sources analyzed: {{ prediction.source_count }}
🤖 Model: {{ prediction.model_used }}
⏰ Generated in {{ prediction.generation_time_seconds }}s
""")


class Notifier:
    """Sends prediction notifications through configured channels."""

    def __init__(self):
        self.settings = get_settings()
        self._apprise = None

    def _get_apprise(self):
        if self._apprise is None:
            import apprise

            self._apprise = apprise.Apprise()
            for url in self.settings.notification_url_list:
                self._apprise.add(url)
        return self._apprise

    async def send(self, prediction: StockPrediction) -> bool:
        """Send a prediction notification to all configured channels."""
        if not self.settings.notification_url_list:
            logger.info("no_notification_urls_configured")
            return False

        message = NOTIFICATION_TEMPLATE.render(prediction=prediction)
        title = (
            f"{'🟢' if prediction.direction.value == 'bullish' else '🔴' if prediction.direction.value == 'bearish' else '⚪'} "
            f"{prediction.ticker}: {prediction.direction.value.upper()} "
            f"({prediction.confidence_pct:.0f}% confidence)"
        )

        try:
            ap = self._get_apprise()
            result = ap.notify(title=title, body=message, notify_type="info")
            logger.info(
                "notification_sent",
                ticker=prediction.ticker,
                channels=len(self.settings.notification_url_list),
                success=result,
            )
            return result
        except Exception as e:
            logger.error("notification_failed", error=str(e))
            return False

    def format_console(self, prediction: StockPrediction) -> str:
        """Format prediction for console output."""
        return NOTIFICATION_TEMPLATE.render(prediction=prediction)

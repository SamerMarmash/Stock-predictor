"""Professional day trading strategies applied to technical signals.

Implements strategies commonly used by professional traders:
1. Trend Following - ride momentum with moving average confirmation
2. Mean Reversion - fade overextended moves
3. Breakout Trading - enter on range expansion
4. Scalping Setup - fast EMA crossovers + volume surge
5. VWAP Strategy - institutional support/resistance levels
6. Divergence Trading - price vs indicator divergence
7. Gap and Go / Gap Fade - opening gap plays
8. Options Flow - put/call ratio signal
"""

from __future__ import annotations

import pandas as pd

from src.core.models import Direction, StrategySignal, TechnicalSignal


class TradingStrategyEngine:
    """Applies professional trading strategies to technical data."""

    def evaluate(
        self,
        signals: list[TechnicalSignal],
        current_data: dict,
        history: pd.DataFrame,
        options_data: dict | None = None,
    ) -> list[StrategySignal]:
        """Run all strategies and return their combined assessment."""
        results = []
        signal_map = {s.indicator_name: s for s in signals}

        results.append(self._trend_following(signal_map, signals))
        results.append(self._mean_reversion(signal_map, signals))
        results.append(self._breakout_strategy(signal_map, signals, history))
        results.append(self._scalping_setup(signal_map, signals))
        results.append(self._vwap_strategy(signal_map, signals))
        results.append(self._divergence_trading(signal_map, signals, history))
        results.append(self._gap_strategy(current_data, history))

        if options_data and options_data.get("available"):
            results.append(self._options_flow_strategy(options_data))

        return [r for r in results if r is not None]

    def _trend_following(
        self, signal_map: dict, signals: list[TechnicalSignal]
    ) -> StrategySignal:
        """Trend following: align with the dominant trend using MA + ADX confirmation."""
        trend_signals = []
        supporting = []

        for name in ["SMA 20/50 Crossover", "EMA 9/21 Crossover", "Price vs SMA200", "ADX (14)", "Ichimoku Cloud"]:
            if name in signal_map:
                trend_signals.append(signal_map[name])
                supporting.append(signal_map[name])

        if not trend_signals:
            return StrategySignal(
                strategy_name="Trend Following",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning="Insufficient trend data available",
                supporting_indicators=supporting,
            )

        bullish = sum(1 for s in trend_signals if s.signal == Direction.BULLISH)
        bearish = sum(1 for s in trend_signals if s.signal == Direction.BEARISH)
        total = len(trend_signals)

        if bullish > bearish:
            direction = Direction.BULLISH
            confidence = bullish / total
        elif bearish > bullish:
            direction = Direction.BEARISH
            confidence = bearish / total
        else:
            direction = Direction.NEUTRAL
            confidence = 0.4

        # Boost confidence if ADX confirms strong trend
        adx = signal_map.get("ADX (14)")
        if adx and adx.value > 30:
            confidence = min(1.0, confidence * 1.2)

        return StrategySignal(
            strategy_name="Trend Following",
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=f"{bullish}/{total} trend indicators bullish. {'Strong' if adx and adx.value > 30 else 'Moderate'} trend strength.",
            supporting_indicators=supporting,
        )

    def _mean_reversion(
        self, signal_map: dict, signals: list[TechnicalSignal]
    ) -> StrategySignal:
        """Mean reversion: fade extremes when RSI/Stochastic show overbought/oversold."""
        reversion_signals = []
        supporting = []

        for name in ["RSI (14)", "Stochastic (14,3)", "Bollinger Bands", "Williams %R"]:
            if name in signal_map:
                reversion_signals.append(signal_map[name])
                supporting.append(signal_map[name])

        if not reversion_signals:
            return StrategySignal(
                strategy_name="Mean Reversion",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning="Insufficient oscillator data",
                supporting_indicators=supporting,
            )

        rsi = signal_map.get("RSI (14)")
        extreme = False
        if rsi:
            if rsi.value > 75 or rsi.value < 25:
                extreme = True

        bullish = sum(1 for s in reversion_signals if s.signal == Direction.BULLISH)
        bearish = sum(1 for s in reversion_signals if s.signal == Direction.BEARISH)
        total = len(reversion_signals)

        if bullish > bearish:
            direction = Direction.BULLISH
            confidence = bullish / total
        elif bearish > bullish:
            direction = Direction.BEARISH
            confidence = bearish / total
        else:
            direction = Direction.NEUTRAL
            confidence = 0.35

        if extreme:
            confidence = min(1.0, confidence * 1.3)

        return StrategySignal(
            strategy_name="Mean Reversion",
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=f"{'Extreme' if extreme else 'Moderate'} readings: {bullish}/{total} oscillators suggest reversal.",
            supporting_indicators=supporting,
        )

    def _breakout_strategy(
        self, signal_map: dict, signals: list[TechnicalSignal], history: pd.DataFrame
    ) -> StrategySignal:
        """Breakout: detect range expansion with volume confirmation."""
        supporting = []

        # Check for volume spike
        vol_spike = signal_map.get("Volume Spike")
        has_volume = vol_spike is not None and vol_spike.value > 1.5

        # Check if price is near BB extremes (potential breakout)
        bb = signal_map.get("Bollinger Bands")
        atr = signal_map.get("ATR (14)")

        if vol_spike:
            supporting.append(vol_spike)
        if bb:
            supporting.append(bb)
        if atr:
            supporting.append(atr)

        if not history.empty and len(history) >= 20:
            recent_range = float(history["High"].tail(20).max() - history["Low"].tail(20).min())
            current = float(history["Close"].iloc[-1])
            high_20 = float(history["High"].tail(20).max())
            low_20 = float(history["Low"].tail(20).min())

            near_high = (high_20 - current) / recent_range < 0.1
            near_low = (current - low_20) / recent_range < 0.1

            if near_high and has_volume:
                return StrategySignal(
                    strategy_name="Breakout Trading",
                    direction=Direction.BULLISH,
                    confidence=0.65 if has_volume else 0.45,
                    reasoning=f"Price near 20-period high with {'strong' if has_volume else 'normal'} volume. Potential upside breakout.",
                    supporting_indicators=supporting,
                )
            elif near_low and has_volume:
                return StrategySignal(
                    strategy_name="Breakout Trading",
                    direction=Direction.BEARISH,
                    confidence=0.6 if has_volume else 0.4,
                    reasoning=f"Price near 20-period low with {'strong' if has_volume else 'normal'} volume. Potential breakdown.",
                    supporting_indicators=supporting,
                )

        return StrategySignal(
            strategy_name="Breakout Trading",
            direction=Direction.NEUTRAL,
            confidence=0.35,
            reasoning="No clear breakout setup detected.",
            supporting_indicators=supporting,
        )

    def _scalping_setup(
        self, signal_map: dict, signals: list[TechnicalSignal]
    ) -> StrategySignal:
        """Scalping: fast EMA crossover + RSI confirmation + volume."""
        supporting = []
        ema = signal_map.get("EMA 9/21 Crossover")
        rsi = signal_map.get("RSI (14)")
        macd = signal_map.get("MACD")

        for s in [ema, rsi, macd]:
            if s:
                supporting.append(s)

        if not ema:
            return StrategySignal(
                strategy_name="Scalping Setup",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning="EMA data not available for scalping assessment.",
                supporting_indicators=supporting,
            )

        # Scalp long: EMA bullish + RSI not overbought + MACD positive
        direction = ema.signal
        confidence = 0.5

        if rsi and ((direction == Direction.BULLISH and rsi.value < 65) or
                    (direction == Direction.BEARISH and rsi.value > 35)):
            confidence += 0.15

        if macd and macd.signal == direction:
            confidence += 0.15

        return StrategySignal(
            strategy_name="Scalping Setup",
            direction=direction,
            confidence=round(min(1.0, confidence), 3),
            reasoning=f"EMA crossover {direction.value}. RSI: {f'{rsi.value:.0f}' if rsi else 'N/A'}, MACD: {macd.signal.value if macd else 'N/A'}.",
            supporting_indicators=supporting,
        )

    def _vwap_strategy(
        self, signal_map: dict, signals: list[TechnicalSignal]
    ) -> StrategySignal:
        """VWAP: institutional level trading - bounce or rejection off VWAP."""
        vwap = signal_map.get("VWAP")
        supporting = [vwap] if vwap else []

        if not vwap:
            return StrategySignal(
                strategy_name="VWAP Strategy",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning="VWAP data not available.",
                supporting_indicators=supporting,
            )

        obv = signal_map.get("OBV Trend")
        if obv:
            supporting.append(obv)

        confidence = 0.5
        if obv and obv.signal == vwap.signal:
            confidence = 0.65

        return StrategySignal(
            strategy_name="VWAP Strategy",
            direction=vwap.signal,
            confidence=round(confidence, 3),
            reasoning=f"Price {'above' if vwap.signal == Direction.BULLISH else 'below'} VWAP. Volume trend {'confirms' if obv and obv.signal == vwap.signal else 'inconclusive'}.",
            supporting_indicators=supporting,
        )

    def _divergence_trading(
        self, signal_map: dict, signals: list[TechnicalSignal], history: pd.DataFrame
    ) -> StrategySignal:
        """Divergence: detect price/RSI or price/MACD divergence."""
        supporting = []
        rsi = signal_map.get("RSI (14)")
        macd = signal_map.get("MACD")

        if rsi:
            supporting.append(rsi)
        if macd:
            supporting.append(macd)

        if not history.empty and len(history) >= 10:
            close = history["Close"]
            price_trend = float(close.iloc[-1]) - float(close.iloc[-10])

            rsi_divergence = False
            if rsi:
                # Bearish divergence: price up but RSI trending down
                if price_trend > 0 and rsi.value < 55:
                    rsi_divergence = True
                    return StrategySignal(
                        strategy_name="Divergence Trading",
                        direction=Direction.BEARISH,
                        confidence=0.6,
                        reasoning=f"Bearish divergence: price rising but RSI ({rsi.value:.0f}) weakening.",
                        supporting_indicators=supporting,
                    )
                # Bullish divergence: price down but RSI trending up
                elif price_trend < 0 and rsi.value > 45:
                    rsi_divergence = True
                    return StrategySignal(
                        strategy_name="Divergence Trading",
                        direction=Direction.BULLISH,
                        confidence=0.6,
                        reasoning=f"Bullish divergence: price falling but RSI ({rsi.value:.0f}) strengthening.",
                        supporting_indicators=supporting,
                    )

        return StrategySignal(
            strategy_name="Divergence Trading",
            direction=Direction.NEUTRAL,
            confidence=0.35,
            reasoning="No clear divergence pattern detected.",
            supporting_indicators=supporting,
        )

    def _gap_strategy(self, current_data: dict, history: pd.DataFrame) -> StrategySignal:
        """Gap analysis: identify gap up/down and whether to trade with or against it."""
        supporting = []

        prev_close = current_data.get("previous_close", 0)
        open_price = current_data.get("open", 0)

        if prev_close == 0 or open_price == 0:
            return StrategySignal(
                strategy_name="Gap Strategy",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning="Insufficient data for gap analysis.",
                supporting_indicators=supporting,
            )

        gap_pct = (open_price - prev_close) / prev_close * 100

        if abs(gap_pct) < 0.5:
            return StrategySignal(
                strategy_name="Gap Strategy",
                direction=Direction.NEUTRAL,
                confidence=0.3,
                reasoning=f"No significant gap ({gap_pct:+.2f}%).",
                supporting_indicators=supporting,
            )

        current = current_data.get("current_price", open_price)
        gap_filled = (gap_pct > 0 and current < prev_close) or (gap_pct < 0 and current > prev_close)

        if gap_filled:
            direction = Direction.BEARISH if gap_pct > 0 else Direction.BULLISH
            return StrategySignal(
                strategy_name="Gap Strategy",
                direction=direction,
                confidence=0.55,
                reasoning=f"Gap {gap_pct:+.2f}% has been filled - gap fade played out.",
                supporting_indicators=supporting,
            )

        # Gap and go: if gap holds, trend continues
        if gap_pct > 1:
            return StrategySignal(
                strategy_name="Gap Strategy",
                direction=Direction.BULLISH,
                confidence=0.55,
                reasoning=f"Gap up {gap_pct:+.2f}% holding - potential gap-and-go continuation.",
                supporting_indicators=supporting,
            )
        elif gap_pct < -1:
            return StrategySignal(
                strategy_name="Gap Strategy",
                direction=Direction.BEARISH,
                confidence=0.55,
                reasoning=f"Gap down {gap_pct:+.2f}% holding - potential continuation lower.",
                supporting_indicators=supporting,
            )

        return StrategySignal(
            strategy_name="Gap Strategy",
            direction=Direction.NEUTRAL,
            confidence=0.35,
            reasoning=f"Small gap ({gap_pct:+.2f}%) - no clear edge.",
            supporting_indicators=supporting,
        )

    def _options_flow_strategy(self, options_data: dict) -> StrategySignal:
        """Options flow: use put/call ratio and unusual volume as a signal."""
        pc_ratio = options_data.get("put_call_ratio", 1.0)
        signal = options_data.get("signal", "neutral")

        if signal == "bullish":
            direction = Direction.BULLISH
            confidence = min(0.7, 0.5 + (1.0 - pc_ratio) * 0.5)
            reasoning = f"Put/call ratio {pc_ratio:.2f} shows bullish options flow (more calls than puts)."
        elif signal == "bearish":
            direction = Direction.BEARISH
            confidence = min(0.7, 0.5 + (pc_ratio - 1.0) * 0.3)
            reasoning = f"Put/call ratio {pc_ratio:.2f} shows bearish options flow (heavy put buying)."
        else:
            direction = Direction.NEUTRAL
            confidence = 0.4
            reasoning = f"Put/call ratio {pc_ratio:.2f} is neutral."

        return StrategySignal(
            strategy_name="Options Flow",
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=reasoning,
        )

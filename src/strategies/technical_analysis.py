"""Technical analysis indicators used by professional day traders.

All indicators are implemented with pure numpy/pandas - no external TA library needed.
- Moving averages (SMA, EMA, VWAP)
- Momentum (RSI, MACD, Stochastic, ADX, Williams %R)
- Volume analysis (OBV, volume profile, accumulation/distribution)
- Volatility (Bollinger Bands, ATR)
- Trend (ADX, Ichimoku Cloud)
- Support/Resistance (Pivot Points)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.models import Direction, TechnicalSignal


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3):
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(smooth).mean()
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def _acc_dist(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    return (mfm * volume).cumsum()


def _bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(lower=0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(lower=0)

    atr_vals = _atr(high, low, close, window)
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr_vals.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr_vals.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.rolling(window).mean()
    return adx_val, plus_di, minus_di


class TechnicalAnalyzer:
    """Runs a comprehensive suite of technical indicators on price data."""

    def analyze(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        """Run all technical indicators and return signals."""
        if df.empty or len(df) < 20:
            return []

        signals = []
        signals.extend(self._moving_averages(df))
        signals.extend(self._momentum_indicators(df))
        signals.extend(self._volume_indicators(df))
        signals.extend(self._volatility_indicators(df))
        signals.extend(self._trend_indicators(df))
        signals.extend(self._support_resistance(df))
        return signals

    def _moving_averages(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        close = df["Close"]
        current = float(close.iloc[-1])

        # SMA crossovers (Golden Cross / Death Cross)
        if len(df) >= 50:
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])

            if sma_20 > sma_50:
                signals.append(TechnicalSignal(
                    indicator_name="SMA 20/50 Crossover",
                    value=sma_20 - sma_50,
                    signal=Direction.BULLISH,
                    strength=min(1.0, abs(sma_20 - sma_50) / current * 10),
                    description=f"SMA20 ({sma_20:.2f}) above SMA50 ({sma_50:.2f}) - bullish crossover",
                ))
            else:
                signals.append(TechnicalSignal(
                    indicator_name="SMA 20/50 Crossover",
                    value=sma_20 - sma_50,
                    signal=Direction.BEARISH,
                    strength=min(1.0, abs(sma_20 - sma_50) / current * 10),
                    description=f"SMA20 ({sma_20:.2f}) below SMA50 ({sma_50:.2f}) - bearish crossover",
                ))

        # EMA 9/21 (fast scalping signal)
        if len(df) >= 21:
            ema_9 = float(close.ewm(span=9).mean().iloc[-1])
            ema_21 = float(close.ewm(span=21).mean().iloc[-1])
            direction = Direction.BULLISH if ema_9 > ema_21 else Direction.BEARISH
            signals.append(TechnicalSignal(
                indicator_name="EMA 9/21 Crossover",
                value=ema_9 - ema_21,
                signal=direction,
                strength=min(1.0, abs(ema_9 - ema_21) / current * 15),
                description=f"EMA9 ({ema_9:.2f}) vs EMA21 ({ema_21:.2f})",
            ))

        # Price vs 200-day SMA (major trend)
        if len(df) >= 200:
            sma_200 = float(close.rolling(200).mean().iloc[-1])
            direction = Direction.BULLISH if current > sma_200 else Direction.BEARISH
            pct_away = abs(current - sma_200) / sma_200
            signals.append(TechnicalSignal(
                indicator_name="Price vs SMA200",
                value=current - sma_200,
                signal=direction,
                strength=min(1.0, pct_away * 5),
                description=f"Price {'above' if current > sma_200 else 'below'} 200-SMA ({sma_200:.2f})",
            ))

        # VWAP (if volume data available)
        if "Volume" in df.columns and len(df) > 1:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            cumulative_vp = (typical_price * df["Volume"]).cumsum()
            cumulative_vol = df["Volume"].cumsum()
            vwap = float((cumulative_vp / cumulative_vol).iloc[-1])
            direction = Direction.BULLISH if current > vwap else Direction.BEARISH
            signals.append(TechnicalSignal(
                indicator_name="VWAP",
                value=vwap,
                signal=direction,
                strength=min(1.0, abs(current - vwap) / current * 10),
                description=f"Price {'above' if current > vwap else 'below'} VWAP ({vwap:.2f})",
            ))

        return signals

    def _momentum_indicators(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        close = df["Close"]

        # RSI
        rsi_series = _rsi(close, 14)
        rsi_val = float(rsi_series.iloc[-1])
        if np.isnan(rsi_val):
            rsi_val = 50.0
        if rsi_val > 70:
            direction, desc = Direction.BEARISH, f"RSI {rsi_val:.1f} - overbought, potential reversal"
        elif rsi_val < 30:
            direction, desc = Direction.BULLISH, f"RSI {rsi_val:.1f} - oversold, potential bounce"
        else:
            direction, desc = Direction.NEUTRAL, f"RSI {rsi_val:.1f} - neutral zone"
        signals.append(TechnicalSignal(
            indicator_name="RSI (14)",
            value=rsi_val,
            signal=direction,
            strength=abs(rsi_val - 50) / 50,
            description=desc,
        ))

        # MACD
        macd_line, macd_signal, macd_hist = _macd(close)
        ml = float(macd_line.iloc[-1])
        ms = float(macd_signal.iloc[-1])
        mh = float(macd_hist.iloc[-1])
        if not (np.isnan(ml) or np.isnan(ms)):
            direction = Direction.BULLISH if ml > ms else Direction.BEARISH
            signals.append(TechnicalSignal(
                indicator_name="MACD",
                value=mh,
                signal=direction,
                strength=min(1.0, abs(mh) / (abs(ml) + 0.001)),
                description=f"MACD line {'above' if ml > ms else 'below'} signal (histogram: {mh:.4f})",
            ))

        # Stochastic Oscillator
        k, d = _stochastic(df["High"], df["Low"], close)
        k_val = float(k.iloc[-1])
        d_val = float(d.iloc[-1])
        if not (np.isnan(k_val) or np.isnan(d_val)):
            if k_val > 80 and d_val > 80:
                direction = Direction.BEARISH
            elif k_val < 20 and d_val < 20:
                direction = Direction.BULLISH
            else:
                direction = Direction.BULLISH if k_val > d_val else Direction.BEARISH
            signals.append(TechnicalSignal(
                indicator_name="Stochastic (14,3)",
                value=k_val,
                signal=direction,
                strength=abs(k_val - 50) / 50,
                description=f"Stochastic %K={k_val:.1f}, %D={d_val:.1f}",
            ))

        # Williams %R
        wr = _williams_r(df["High"], df["Low"], close)
        wr_val = float(wr.iloc[-1])
        if not np.isnan(wr_val):
            if wr_val > -20:
                direction = Direction.BEARISH
            elif wr_val < -80:
                direction = Direction.BULLISH
            else:
                direction = Direction.NEUTRAL
            signals.append(TechnicalSignal(
                indicator_name="Williams %R",
                value=wr_val,
                signal=direction,
                strength=abs(wr_val + 50) / 50,
                description=f"Williams %R: {wr_val:.1f}",
            ))

        return signals

    def _volume_indicators(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        if "Volume" not in df.columns:
            return signals

        close = df["Close"]
        volume = df["Volume"]

        # On-Balance Volume trend
        obv = _obv(close, volume)
        if len(obv) >= 20:
            obv_sma = obv.rolling(20).mean()
            current_obv = float(obv.iloc[-1])
            sma_obv = float(obv_sma.iloc[-1])
            if not np.isnan(sma_obv):
                direction = Direction.BULLISH if current_obv > sma_obv else Direction.BEARISH
                signals.append(TechnicalSignal(
                    indicator_name="OBV Trend",
                    value=current_obv,
                    signal=direction,
                    strength=0.6,
                    description=f"OBV {'rising above' if current_obv > sma_obv else 'falling below'} its 20-period average",
                ))

        # Volume relative to average (unusual activity detection)
        avg_vol = float(volume.rolling(20).mean().iloc[-1])
        curr_vol = float(volume.iloc[-1])
        if avg_vol > 0 and not np.isnan(avg_vol):
            vol_ratio = curr_vol / avg_vol
            if vol_ratio > 2.0:
                signals.append(TechnicalSignal(
                    indicator_name="Volume Spike",
                    value=vol_ratio,
                    signal=Direction.NEUTRAL,
                    strength=min(1.0, vol_ratio / 4),
                    description=f"Volume is {vol_ratio:.1f}x average - unusual activity detected",
                ))

        # Accumulation/Distribution
        ad = _acc_dist(df["High"], df["Low"], close, volume)
        if len(ad) >= 10:
            ad_now = float(ad.iloc[-1])
            ad_prev = float(ad.iloc[-10])
            if not (np.isnan(ad_now) or np.isnan(ad_prev)):
                ad_trend = ad_now - ad_prev
                direction = Direction.BULLISH if ad_trend > 0 else Direction.BEARISH
                signals.append(TechnicalSignal(
                    indicator_name="Accumulation/Distribution",
                    value=ad_trend,
                    signal=direction,
                    strength=0.55,
                    description=f"A/D line {'accumulating' if ad_trend > 0 else 'distributing'} over last 10 periods",
                ))

        return signals

    def _volatility_indicators(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        close = df["Close"]
        current = float(close.iloc[-1])

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
        upper = float(bb_upper.iloc[-1])
        lower = float(bb_lower.iloc[-1])
        mid = float(bb_mid.iloc[-1])
        if not (np.isnan(upper) or np.isnan(lower) or np.isnan(mid)):
            bb_width = (upper - lower) / mid if mid != 0 else 0

            if current > upper:
                direction, desc = Direction.BEARISH, f"Price above upper BB ({upper:.2f}) - overbought"
            elif current < lower:
                direction, desc = Direction.BULLISH, f"Price below lower BB ({lower:.2f}) - oversold"
            else:
                band_range = upper - lower
                position = (current - lower) / band_range if band_range > 0 else 0.5
                direction = Direction.BULLISH if position < 0.4 else Direction.BEARISH if position > 0.6 else Direction.NEUTRAL
                desc = f"Price at {position:.0%} of BB range (width: {bb_width:.3f})"

            signals.append(TechnicalSignal(
                indicator_name="Bollinger Bands",
                value=current,
                signal=direction,
                strength=0.6 if direction != Direction.NEUTRAL else 0.3,
                description=desc,
            ))

        # ATR for volatility context
        atr_series = _atr(df["High"], df["Low"], close)
        atr_val = float(atr_series.iloc[-1])
        if not np.isnan(atr_val) and current > 0:
            atr_pct = atr_val / current * 100
            signals.append(TechnicalSignal(
                indicator_name="ATR (14)",
                value=atr_val,
                signal=Direction.NEUTRAL,
                strength=min(1.0, atr_pct / 5),
                description=f"ATR: {atr_val:.2f} ({atr_pct:.1f}% of price) - {'high' if atr_pct > 3 else 'moderate' if atr_pct > 1.5 else 'low'} volatility",
            ))

        return signals

    def _trend_indicators(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        close = df["Close"]

        # ADX - Average Directional Index (trend strength)
        adx_val_series, plus_di, minus_di = _adx(df["High"], df["Low"], close)
        adx_v = float(adx_val_series.iloc[-1])
        di_p = float(plus_di.iloc[-1])
        di_m = float(minus_di.iloc[-1])
        if not (np.isnan(adx_v) or np.isnan(di_p) or np.isnan(di_m)):
            trending = adx_v > 25
            direction = Direction.BULLISH if di_p > di_m else Direction.BEARISH

            signals.append(TechnicalSignal(
                indicator_name="ADX (14)",
                value=adx_v,
                signal=direction if trending else Direction.NEUTRAL,
                strength=min(1.0, adx_v / 50),
                description=f"ADX: {adx_v:.1f} ({'strong trend' if adx_v > 40 else 'trending' if trending else 'weak/no trend'}), DI+: {di_p:.1f}, DI-: {di_m:.1f}",
            ))

        # Ichimoku Cloud
        if len(df) >= 52:
            high = df["High"]
            low = df["Low"]
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            current = float(close.iloc[-1])
            t = float(tenkan.iloc[-1])
            k = float(kijun.iloc[-1])

            if not (np.isnan(t) or np.isnan(k)):
                if current > t and t > k:
                    direction = Direction.BULLISH
                    strength = 0.7
                elif current < t and t < k:
                    direction = Direction.BEARISH
                    strength = 0.7
                else:
                    direction = Direction.NEUTRAL
                    strength = 0.3

                signals.append(TechnicalSignal(
                    indicator_name="Ichimoku Cloud",
                    value=t - k,
                    signal=direction,
                    strength=strength,
                    description=f"Tenkan: {t:.2f}, Kijun: {k:.2f}, Price: {current:.2f}",
                ))

        return signals

    def _support_resistance(self, df: pd.DataFrame) -> list[TechnicalSignal]:
        signals = []
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        current = float(close.iloc[-1])

        # Pivot points (classic)
        prev_high = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
        prev_low = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])

        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)

        # Determine proximity to support/resistance
        nearest_resistance = min(r1, r2, key=lambda x: abs(x - current) if x > current else float("inf"))
        nearest_support = min(s1, s2, key=lambda x: abs(x - current) if x < current else float("inf"))

        if abs(current - nearest_support) < abs(current - nearest_resistance):
            direction = Direction.BULLISH
            desc = f"Near support at {nearest_support:.2f} (pivot: {pivot:.2f})"
        else:
            direction = Direction.BEARISH
            desc = f"Near resistance at {nearest_resistance:.2f} (pivot: {pivot:.2f})"

        signals.append(TechnicalSignal(
            indicator_name="Pivot Points",
            value=pivot,
            signal=direction,
            strength=0.5,
            description=desc,
        ))

        return signals

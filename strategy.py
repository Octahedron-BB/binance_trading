"""Shared strategy indicator and signal definitions for live trading and backtesting."""

from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the dual-EMA + MACD trend-following strategy."""

    ema_fast: int = 30
    ema_slow: int = 200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Supported: "ema30_macd" | "ema200_regime" | "ema200_cross" | "ema200_macd" | "ema200_macd_both_exit"
    signal_mode: str = "ema30_macd"


def add_indicators(df: pd.DataFrame, config: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    """Compute EMA and MACD histogram columns matching the live bot."""
    out = df.copy()
    out.ta.ema(length=config.ema_fast, append=True)
    out.ta.ema(length=config.ema_slow, append=True)
    out.ta.macd(
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
        append=True,
    )

    ema_fast_col = f"EMA_{config.ema_fast}"
    ema_slow_col = f"EMA_{config.ema_slow}"
    hist_col = f"MACDh_{config.macd_fast}_{config.macd_slow}_{config.macd_signal}"

    out.rename(
        columns={
            ema_fast_col: "ema30",
            ema_slow_col: "ema200",
            hist_col: "hist",
        },
        inplace=True,
    )
    out.dropna(subset=["ema200", "hist"], inplace=True)
    return out


def check_signal_rows(
    current: pd.Series,
    prev: pd.Series,
    signal_mode: str = "ema30_macd",
) -> str:
    """Evaluate a single bar transition against the previous bar."""
    if signal_mode == "ema200_regime":
        if current["close"] > current["ema200"]:
            return "LONG_ENTRY"
        if current["close"] < current["ema200"]:
            return "LONG_EXIT"
        return "HOLD"

    if signal_mode == "ema200_cross":
        if current["close"] < current["ema200"]:
            return "LONG_EXIT"
        if prev["close"] <= prev["ema200"] and current["close"] > current["ema200"]:
            return "LONG_ENTRY"
        return "HOLD"

    if signal_mode == "ema200_macd":
        # Direct removal of EMA30 from original logic: require close > EMA200 and hist > 0, exit on either failing.
        if current["hist"] < 0 or current["close"] < current["ema200"]:
            return "LONG_EXIT"
        cond_a = (prev["hist"] <= 0 and current["hist"] > 0) and (current["close"] > current["ema200"])
        cond_b = (prev["close"] <= prev["ema200"] and current["close"] > current["ema200"]) and (
            current["hist"] > 0
        )
        if cond_a or cond_b:
            return "LONG_ENTRY"
        return "HOLD"

    if signal_mode == "ema200_macd_both_exit":
        # Enter when both trend and momentum are bullish; exit only when both turn bearish.
        if current["close"] > current["ema200"] and current["hist"] > 0:
            return "LONG_ENTRY"
        if current["close"] < current["ema200"] and current["hist"] < 0:
            return "LONG_EXIT"
        return "HOLD"

    if signal_mode != "ema30_macd":
        raise ValueError(f"Unsupported signal_mode: {signal_mode}")

    macro_bull = current["close"] > current["ema200"]

    # 進場 A: MACD 金叉 + 價高於 EMA30
    cond_a = (prev["hist"] <= 0 and current["hist"] > 0) and (current["close"] > current["ema30"])
    # 進場 B: 價穿 EMA30 + MACD 柱狀 > 0
    cond_b = (prev["close"] <= prev["ema30"] and current["close"] > current["ema30"]) and (
        current["hist"] > 0
    )

    # 出場: MACD 轉負 或 跌破 EMA30
    if current["hist"] < 0 or current["close"] < current["ema30"]:
        return "LONG_EXIT"

    if (cond_a or cond_b) and macro_bull:
        return "LONG_ENTRY"

    return "HOLD"


def check_signals(df: pd.DataFrame, signal_mode: str = "ema30_macd") -> str:
    """Evaluate the latest completed bar in an indicator-enriched DataFrame."""
    if len(df) < 2:
        return "HOLD"
    return check_signal_rows(df.iloc[-1], df.iloc[-2], signal_mode=signal_mode)


def generate_signal_series(df: pd.DataFrame, signal_mode: str = "ema30_macd") -> pd.Series:
    """Produce bar-by-bar signals identical to calling check_signals on prefixes."""
    signals = ["HOLD"] * len(df)
    for i in range(1, len(df)):
        signals[i] = check_signal_rows(df.iloc[i], df.iloc[i - 1], signal_mode=signal_mode)
    return pd.Series(signals, index=df.index, name="signal")

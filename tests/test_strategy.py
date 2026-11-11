"""Verification tests for strategy parity, EMA200 modes, and zero-lookahead causality."""

import datetime as dt
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import bot
from backtest import AssetSpec, run_portfolio_backtest
from strategy import StrategyConfig, add_indicators, check_signals, generate_signal_series


def _legacy_check_signals(df: pd.DataFrame) -> str:
    """Original bot.py signal logic kept here solely as a regression oracle."""
    current = df.iloc[-1]
    prev = df.iloc[-2]
    macro_bull = current["close"] > current["ema200"]
    cond_a = (prev["hist"] <= 0 and current["hist"] > 0) and (current["close"] > current["ema30"])
    cond_b = (prev["close"] <= prev["ema30"] and current["close"] > current["ema30"]) and (
        current["hist"] > 0
    )
    if current["hist"] < 0 or current["close"] < current["ema30"]:
        return "LONG_EXIT"
    if (cond_a or cond_b) and macro_bull:
        return "LONG_ENTRY"
    return "HOLD"


def _make_synthetic_ohlcv(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    returns = rng.normal(0.001, 0.02, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_px = np.roll(close, 1)
    open_px[0] = 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_px,
            "high": np.maximum(open_px, close) * 1.01,
            "low": np.minimum(open_px, close) * 0.99,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


class StrategyAndBacktestTests(unittest.TestCase):
    def test_signal_parity_with_legacy_bot(self) -> None:
        raw = _make_synthetic_ohlcv(340, seed=42)
        enriched = add_indicators(raw, StrategyConfig())
        series = generate_signal_series(enriched)

        for i in range(2, len(enriched) + 1):
            sub = enriched.iloc[:i]
            expected = _legacy_check_signals(sub)
            self.assertEqual(check_signals(sub), expected)
            self.assertEqual(series.iloc[i - 1], expected)

    def test_ema200_modes_and_total_equity_target(self) -> None:
        raw = _make_synthetic_ohlcv(340, seed=5)
        enriched = add_indicators(raw, StrategyConfig(signal_mode="ema200_regime"))
        regime = generate_signal_series(enriched, signal_mode="ema200_regime")
        cross = generate_signal_series(enriched, signal_mode="ema200_cross")

        # Regime emits LONG_ENTRY whenever close > ema200; cross only on transition.
        self.assertTrue(((regime == "LONG_ENTRY") | (regime == "LONG_EXIT") | (regime == "HOLD")).all())
        self.assertLessEqual((cross == "LONG_ENTRY").sum(), (regime == "LONG_ENTRY").sum())

        for mode in ["ema200_macd", "ema200_macd_both_exit"]:
            sig_s = generate_signal_series(enriched, signal_mode=mode)
            self.assertTrue(((sig_s == "LONG_ENTRY") | (sig_s == "LONG_EXIT") | (sig_s == "HOLD")).all())

        assets = [
            AssetSpec("A/USDT", "A=F", "A/USDT", 0.5),
            AssetSpec("B/USDT", "B=F", "B/USDT", 0.5),
        ]
        res = run_portfolio_backtest(
            assets,
            {"A/USDT": raw, "B/USDT": _make_synthetic_ohlcv(340, seed=19)},
            {"A/USDT": raw, "B/USDT": _make_synthetic_ohlcv(340, seed=19)},
            asset_configs={
                "A/USDT": StrategyConfig(signal_mode="ema30_macd"),
                "B/USDT": StrategyConfig(signal_mode="ema200_regime"),
            },
            sizing_mode="total_equity_target",
            deploy_ratio=0.70,
        )
        # With 70% deploy target and two legs, average exposure cannot exceed ~70% significantly.
        self.assertLessEqual(res["metrics"]["avg_exposure_pct"], 80.0)
        self.assertGreater(res["metrics"]["final_equity"], 0.0)

    def test_zero_lookahead_in_backtest(self) -> None:
        sig_df = _make_synthetic_ohlcv(350, seed=11)
        spot_df = _make_synthetic_ohlcv(350, seed=99)
        assets = [AssetSpec("TEST/USDT", "TEST=F", "TEST/USDT", 1.0)]

        base = run_portfolio_backtest(
            assets,
            {"TEST/USDT": sig_df},
            {"TEST/USDT": spot_df},
            sizing_mode="normalized",
        )

        cutoff = sig_df["date"].iloc[300]
        sig_mut = sig_df.copy()
        spot_mut = spot_df.copy()
        sig_mut.loc[sig_mut["date"] > cutoff, ["open", "high", "low", "close"]] *= 5.0
        spot_mut.loc[spot_mut["date"] > cutoff, ["open", "high", "low", "close"]] *= 5.0

        perturbed = run_portfolio_backtest(
            assets,
            {"TEST/USDT": sig_mut},
            {"TEST/USDT": spot_mut},
            sizing_mode="normalized",
        )

        pd.testing.assert_series_equal(
            base["equity"].loc[:cutoff],
            perturbed["equity"].loc[:cutoff],
        )

    def test_bot_equity_sizing_no_duplicate_buy_and_stale_guard(self) -> None:
        class FakeExchange:
            def __init__(self) -> None:
                self.orders: list[tuple] = []
                self.usdt = 1200.0
                self.btc = 0.01  # $800 at 80,000
                self.xaut = 0.0

            def fetch_balance(self) -> dict:
                return {
                    "free": {"USDT": self.usdt, "BTC": self.btc, "XAUT": self.xaut},
                    "total": {"USDT": self.usdt, "BTC": self.btc, "XAUT": self.xaut},
                }

            def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 80_000.0 if symbol == "BTC/USDT" else 4_000.0}

            def create_market_buy_order(self, symbol: str, amount: float, params=None) -> dict:
                self.orders.append(("BUY", symbol, round(amount, 2)))
                self.usdt -= amount
                if symbol == "XAUT/USDT":
                    self.xaut += amount / 4_000.0
                return {"average": 4_000.0}

            def create_market_sell_order(self, symbol: str, amount: float) -> dict:
                self.orders.append(("SELL", symbol, amount))
                return {"average": 80_000.0}

        ex = FakeExchange()
        bot.LAST_PROCESSED_BAR_DATE.clear()

        def _fake_fetch(ticker: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"timestamp": pd.Timestamp("2026-11-09"), "close": 100.0, "ema30": 95.0, "ema200": 90.0, "hist": 1.0},
                    {"timestamp": pd.Timestamp("2026-11-10"), "close": 101.0, "ema30": 96.0, "ema200": 91.0, "hist": 1.5},
                ]
            )

        weekday_utc = dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC)
        with (
            patch.object(bot, "fetch_market_data", side_effect=_fake_fetch),
            patch.object(bot, "check_signals", return_value="LONG_ENTRY"),
            patch.object(bot, "send_telegram_notification"),
        ):
            bot.run_strategy(exchange=ex, now_utc=weekday_utc)
            # BTC already open ($800) so skipped; XAUT targets 2000 * 0.7 * 0.5 = 700.0 USDT
            self.assertEqual(ex.orders, [("BUY", "XAUT/USDT", 700.0)])

            # Re-running with same bar date (e.g. US holiday) must not place any duplicate order
            bot.run_strategy(exchange=ex, now_utc=weekday_utc + dt.timedelta(days=1))
            self.assertEqual(len(ex.orders), 1)

    def test_windowed_backtest_ignores_stale_prestart_signals(self) -> None:
        """Signals older than start_date - 1 day must not queue onto the first window bar."""
        dates = pd.date_range("2024-01-01", periods=280, freq="D")
        closes = np.full(len(dates), 100.0)
        entry_idx = 220  # 2024-08-08
        closes[entry_idx:] = 120.0
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.full(len(dates), 1_000.0),
            }
        )
        entry_sig_date = dates[entry_idx]  # 2024-08-08 -> LONG_ENTRY, then HOLD afterward
        asset = AssetSpec(name="GOLD", signal_ticker="GC=F", spot_symbol="XAUT/USDT", weight=0.5)
        cfg = StrategyConfig(signal_mode="ema200_cross")

        # Window starts 10 days after entry signal -> stale entry must NOT execute
        stale_res = run_portfolio_backtest(
            assets=[asset],
            signal_frames={"GOLD": frame},
            spot_frames={"GOLD": frame},
            strategy_cfg=cfg,
            sizing_mode="total_equity_target",
            deploy_ratio=0.7,
            initial_capital=1_000.0,
            start_date=str((entry_sig_date + pd.Timedelta(days=10)).date()),
            end_date=str(dates[-1].date()),
        )
        self.assertEqual(len(stale_res["trades"]), 0)

        # Window starts the day right after entry signal -> valid next-open execution
        fresh_res = run_portfolio_backtest(
            assets=[asset],
            signal_frames={"GOLD": frame},
            spot_frames={"GOLD": frame},
            strategy_cfg=cfg,
            sizing_mode="total_equity_target",
            deploy_ratio=0.7,
            initial_capital=1_000.0,
            start_date=str((entry_sig_date + pd.Timedelta(days=1)).date()),
            end_date=str(dates[-1].date()),
        )
        self.assertEqual(len(fresh_res["trades"]), 1)
        self.assertEqual(fresh_res["trades"][0].entry_signal_date, str(entry_sig_date.date()))
        self.assertEqual(
            fresh_res["trades"][0].entry_date,
            str((entry_sig_date + pd.Timedelta(days=1)).date()),
        )

    def test_fetch_market_data_drops_future_unclosed_bar(self) -> None:
        """Preserve origin/main unclosed-bar guard: drop bars stamped after UTC today."""
        dates = pd.date_range("2026-01-01", periods=316, freq="D")  # ends 2026-11-12
        closes = np.linspace(100.0, 200.0, len(dates))
        hist_df = pd.DataFrame(
            {
                "Open": closes,
                "High": closes + 1.0,
                "Low": closes - 1.0,
                "Close": closes,
                "Volume": np.full(len(dates), 1_000.0),
            },
            index=pd.DatetimeIndex(dates, name="Date"),
        )

        class FakeTicker:
            def history(self, period: str, interval: str) -> pd.DataFrame:
                return hist_df

        now_utc = dt.datetime(2026, 11, 11, 23, 0, tzinfo=dt.UTC)
        with patch.object(bot.yf, "Ticker", return_value=FakeTicker()):
            out = bot.fetch_market_data("GC=F", now_utc=now_utc)

        self.assertEqual(pd.Timestamp(out["timestamp"].iloc[-1]).date(), dt.date(2026, 11, 11))


if __name__ == "__main__":
    unittest.main()

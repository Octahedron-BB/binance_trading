"""Verification tests for strategy parity, EMA200 modes, and zero-lookahead causality."""

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
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
        bot.PENDING_RETRIES.clear()

        def _fake_fetch(ticker: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"timestamp": pd.Timestamp("2026-11-09"), "close": 100.0, "ema30": 95.0, "ema200": 90.0, "hist": 1.0},
                    {"timestamp": pd.Timestamp("2026-11-10"), "close": 101.0, "ema30": 96.0, "ema200": 91.0, "hist": 1.5},
                ]
            )

        weekday_utc = dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC)
        with (
            TemporaryDirectory() as tmpdir,
            patch.object(bot, "STATE_FILE", Path(tmpdir) / "state.json"),
            patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
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

    @staticmethod
    def _bars_ending(date_str: str) -> pd.DataFrame:
        end = pd.Timestamp(date_str)
        prev = end - pd.Timedelta(days=1)
        return pd.DataFrame(
            [
                {"timestamp": prev, "close": 100.0, "ema30": 95.0, "ema200": 90.0, "hist": 1.0},
                {"timestamp": end, "close": 101.0, "ema30": 96.0, "ema200": 91.0, "hist": 1.5},
            ]
        )

    def test_dust_appreciation_does_not_swallow_buy_and_low_exit_threshold(self) -> None:
        """Small residual >$10 must not swallow a 35% buy; $7 position still exits cleanly."""
        class StubEx:
            def __init__(self, usdt: float, btc_free: float, btc_total: float, xaut_free: float) -> None:
                self.orders: list[tuple] = []
                self.usdt = usdt
                self.btc_free = btc_free
                self.btc_total = btc_total
                self.xaut_free = xaut_free

            def fetch_balance(self) -> dict:
                return {
                    "free": {"USDT": self.usdt, "BTC": self.btc_free, "XAUT": self.xaut_free},
                    "total": {"USDT": self.usdt, "BTC": self.btc_total, "XAUT": self.xaut_free},
                }

            def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 100.0 if symbol == "BTC/USDT" else 100.0}

            def create_market_buy_order(self, symbol: str, amount: float, params=None) -> dict:
                self.orders.append(("BUY", symbol, round(amount, 2)))
                return {"status": "closed", "filled": amount / 100.0, "average": 100.0}

            def create_market_sell_order(self, symbol: str, amount: float) -> dict:
                self.orders.append(("SELL", symbol, round(amount, 4)))
                return {"status": "closed", "filled": amount, "average": 100.0}

        # Case A: BTC has $12 dust (<15% of $350 target), XAUT has $0 -> BTC tops up $350 - $12 = $338
        # Also test BTC rallied + XAUT holds $12? Here total equity = 1988 + 12 = 2000 -> target = 700, 12 < 105 -> buys 688
        ex_buy = StubEx(usdt=1988.0, btc_free=0.12, btc_total=0.12, xaut_free=0.0)
        bot.LAST_PROCESSED_BAR_DATE.clear()
        bot.PENDING_RETRIES.clear()
        day = dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC)
        with (
            TemporaryDirectory() as tmpdir,
            patch.object(bot, "STATE_FILE", Path(tmpdir) / "state.json"),
            patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
            patch.object(bot, "fetch_market_data", return_value=self._bars_ending("2026-11-10")),
            patch.object(bot, "check_signals", return_value="LONG_ENTRY"),
            patch.object(bot, "send_telegram_notification"),
        ):
            bot.run_strategy(exchange=ex_buy, now_utc=day)
            self.assertEqual(ex_buy.orders, [("BUY", "BTC/USDT", 688.0), ("BUY", "XAUT/USDT", 700.0)])

        # Case B: BTC holds $7.0 (between MIN_EXIT_USDT=5.5 and MIN_ORDER_USDT=10) and XAUT is locked in limit order
        ex_exit = StubEx(usdt=1000.0, btc_free=0.07, btc_total=0.07, xaut_free=0.0)
        ex_exit_locked_xaut = 2.0  # $200 total XAUT, $0 free

        def _bal_locked() -> dict:
            return {
                "free": {"USDT": 1000.0, "BTC": 0.07, "XAUT": 0.0},
                "total": {"USDT": 1000.0, "BTC": 0.07, "XAUT": ex_exit_locked_xaut},
            }

        ex_exit.fetch_balance = _bal_locked  # type: ignore[method-assign]
        bot.LAST_PROCESSED_BAR_DATE.clear()
        bot.PENDING_RETRIES.clear()
        sent_msgs: list[str] = []
        with (
            TemporaryDirectory() as tmpdir,
            patch.object(bot, "STATE_FILE", Path(tmpdir) / "state.json"),
            patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
            patch.object(bot, "fetch_market_data", return_value=self._bars_ending("2026-11-10")),
            patch.object(bot, "check_signals", return_value="LONG_EXIT"),
            patch.object(bot, "send_telegram_notification", side_effect=sent_msgs.append),
        ):
            bot.run_strategy(exchange=ex_exit, now_utc=day)
            self.assertEqual(ex_exit.orders, [("SELL", "BTC/USDT", 0.07)])
            self.assertIn("掛單鎖定", sent_msgs[-1])

    def test_persisted_state_prevents_holiday_retrigger_after_restart(self) -> None:
        """Persisted bar date + bar_date < today guard must block holiday stale bars even after restart."""
        class NoopEx:
            def __init__(self) -> None:
                self.orders: list[tuple] = []

            def fetch_balance(self) -> dict:
                return {"free": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}, "total": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}}

            def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 100.0}

            def create_market_buy_order(self, symbol: str, amount: float, params=None) -> dict:
                self.orders.append(("BUY", symbol, amount))
                return {"status": "closed", "filled": 1.0, "average": 100.0}

        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "bot_state.json"
            bot.LAST_PROCESSED_BAR_DATE.clear()
            bot.PENDING_RETRIES.clear()
            ex = NoopEx()

            with (
                patch.object(bot, "STATE_FILE", state_path),
                patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
                patch.object(bot, "fetch_market_data", return_value=self._bars_ending("2026-11-10")),
                patch.object(bot, "check_signals", return_value="HOLD"),
                patch.object(bot, "send_telegram_notification"),
            ):
                bot.run_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC))
                self.assertTrue(state_path.exists())

            # Simulate container restart (memory cleared, reloaded from disk) on a US holiday 2026-11-11
            bot.LAST_PROCESSED_BAR_DATE.clear()
            bot.PENDING_RETRIES.clear()
            with patch.object(bot, "STATE_FILE", state_path):
                bot._init_persisted_state()
                self.assertEqual(bot.LAST_PROCESSED_BAR_DATE.get("BTC=F"), "2026-11-10")

            # Even if signal flipped to LONG_ENTRY in mock, stale bar on holiday must NOT buy
            with (
                patch.object(bot, "STATE_FILE", state_path),
                patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
                patch.object(bot, "fetch_market_data", return_value=self._bars_ending("2026-11-10")),
                patch.object(bot, "check_signals", return_value="LONG_ENTRY"),
                patch.object(bot, "send_telegram_notification"),
            ):
                bot.run_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 11, 23, 0, tzinfo=dt.UTC))
                self.assertEqual(ex.orders, [])

    def test_fetch_retry_and_telegram_notification_on_recovery(self) -> None:
        """Failed 23:00 fetch queues retry; succeeding at 23:15 executes and notifies Telegram."""
        class BuyEx:
            def __init__(self) -> None:
                self.orders: list[tuple] = []

            def fetch_balance(self) -> dict:
                return {"free": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}, "total": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}}

            def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 80_000.0 if symbol == "BTC/USDT" else 4_000.0}

            def create_market_buy_order(self, symbol: str, amount: float, params=None) -> dict:
                self.orders.append(("BUY", symbol, round(amount, 2)))
                return {"status": "closed", "filled": amount / 4_000.0, "average": 4_000.0}

        ex = BuyEx()
        bot.LAST_PROCESSED_BAR_DATE.clear()
        bot.PENDING_RETRIES.clear()
        sent_msgs: list[str] = []
        gc_calls = 0

        def _flaky_fetch(ticker: str) -> pd.DataFrame:
            nonlocal gc_calls
            if ticker == "BTC=F":
                return self._bars_ending("2026-11-10")
            gc_calls += 1
            # Fail all 3 attempts at 23:00 and 1st attempt at 23:15; succeed on 2nd attempt at 23:15
            if gc_calls <= 4:
                raise ConnectionError("Yahoo temporary timeout")
            return self._bars_ending("2026-11-10")

        def _sig_by_len(df: pd.DataFrame, signal_mode: str = "ema30_macd") -> str:
            return "LONG_ENTRY" if signal_mode == "ema200_macd_both_exit" else "HOLD"

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(bot, "STATE_FILE", Path(tmpdir) / "state.json"),
            patch.object(bot, "FETCH_MAX_RETRIES", 3),
            patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
            patch.object(bot, "fetch_market_data", side_effect=_flaky_fetch),
            patch.object(bot, "check_signals", side_effect=_sig_by_len),
            patch.object(bot, "send_telegram_notification", side_effect=sent_msgs.append),
        ):
            bot.run_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC))
            self.assertEqual(ex.orders, [])
            self.assertEqual(bot.PENDING_RETRIES.get("GC=F", {}).get("reason"), "error")
            self.assertIn("抓取失敗", sent_msgs[-1])

            bot.retry_pending_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 10, 23, 15, tzinfo=dt.UTC))
            self.assertEqual(ex.orders, [("BUY", "XAUT/USDT", 350.0)])
            self.assertNotIn("GC=F", bot.PENDING_RETRIES)
            self.assertIn("數據補抓成功並完成執行通知", sent_msgs[-1])
            self.assertIn("重試第 2 次成功", sent_msgs[-1])

    def test_lagging_bar_recovers_in_retry_window_with_telegram_alert(self) -> None:
        """If 23:00 gets yesterday's bar (Yahoo lag) and 23:15 gets today's, trade executes and notifies."""
        class BuyEx:
            def __init__(self) -> None:
                self.orders: list[tuple] = []

            def fetch_balance(self) -> dict:
                return {"free": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}, "total": {"USDT": 1000.0, "BTC": 0.0, "XAUT": 0.0}}

            def fetch_ticker(self, symbol: str) -> dict:
                return {"last": 80_000.0 if symbol == "BTC/USDT" else 4_000.0}

            def create_market_buy_order(self, symbol: str, amount: float, params=None) -> dict:
                self.orders.append(("BUY", symbol, round(amount, 2)))
                return {"status": "closed", "filled": amount / 80_000.0, "average": 80_000.0}

        ex = BuyEx()
        bot.LAST_PROCESSED_BAR_DATE.clear()
        bot.PENDING_RETRIES.clear()
        sent_msgs: list[str] = []
        btc_ready = {"ok": False}

        def _laggy_fetch(ticker: str) -> pd.DataFrame:
            if ticker == "GC=F":
                return self._bars_ending("2026-11-10")
            return self._bars_ending("2026-11-10" if btc_ready["ok"] else "2026-11-09")

        def _sig(df: pd.DataFrame, signal_mode: str = "ema30_macd") -> str:
            return "LONG_ENTRY" if signal_mode == "ema30_macd" else "HOLD"

        with (
            TemporaryDirectory() as tmpdir,
            patch.object(bot, "STATE_FILE", Path(tmpdir) / "state.json"),
            patch.object(bot, "FETCH_RETRY_DELAY", 0.0),
            patch.object(bot, "fetch_market_data", side_effect=_laggy_fetch),
            patch.object(bot, "check_signals", side_effect=_sig),
            patch.object(bot, "send_telegram_notification", side_effect=sent_msgs.append),
        ):
            bot.run_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 10, 23, 0, tzinfo=dt.UTC))
            self.assertEqual(ex.orders, [])
            self.assertEqual(bot.PENDING_RETRIES.get("BTC=F", {}).get("reason"), "lag")

            btc_ready["ok"] = True
            bot.retry_pending_strategy(exchange=ex, now_utc=dt.datetime(2026, 11, 10, 23, 15, tzinfo=dt.UTC))
            self.assertEqual(ex.orders, [("BUY", "BTC/USDT", 350.0)])
            self.assertNotIn("BTC=F", bot.PENDING_RETRIES)
            self.assertIn("數據補抓成功並完成執行通知", sent_msgs[-1])

    def test_default_mainnet_live_mode(self) -> None:
        """Default exchange_client runs on live Mainnet unless USE_TESTNET is enabled."""
        self.assertFalse(bot.USE_TESTNET)
        api_urls = [str(v) for v in bot.exchange_client.urls["api"].values()]
        self.assertTrue(any("api.binance.com" in u for u in api_urls))
        self.assertFalse(any("testnet" in u for u in api_urls))


if __name__ == "__main__":
    unittest.main()

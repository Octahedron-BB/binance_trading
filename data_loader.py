"""Market data fetching and local CSV caching for cross-market backtests."""

from pathlib import Path
import re
import time
import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / ".backtest_cache"
BINANCE_VISION_KLINES = "https://data-api.binance.vision/api/v3/klines"


def _utc_today_naive() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _normalize_yf_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.reset_index(inplace=True)
    date_col = "Date" if "Date" in out.columns else "Datetime"
    idx = pd.to_datetime(out[date_col])
    if idx.dt.tz is not None:
        # Preserve exchange trading calendar date (e.g., CME America/New_York date).
        idx = idx.dt.tz_localize(None)
    out["date"] = idx.dt.normalize()
    out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    cols = ["date", "open", "high", "low", "close", "volume"]
    out = out[cols].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Exclude today's potentially unsettled bar.
    out = out[out["date"] < _utc_today_naive()].reset_index(drop=True)
    return out


def fetch_yfinance_daily(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache closed daily OHLCV bars from Yahoo Finance."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker)
    cache_path = CACHE_DIR / f"yf_{safe_name}_1d.csv"

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached[cached["date"] < _utc_today_naive()].reset_index(drop=True)

    raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

    df = _normalize_yf_dates(raw)
    df.to_csv(cache_path, index=False)
    return df


def fetch_binance_daily(
    symbol: str,
    start_date: str = "2017-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache closed Binance spot daily bars via data-api.binance.vision."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pair = symbol.replace("/", "").upper()
    cache_path = CACHE_DIR / f"binance_{pair}_1d.csv"

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached[cached["date"] < _utc_today_naive()].reset_index(drop=True)

    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(_utc_today_naive().tz_localize("UTC").timestamp() * 1000) - 1
    rows: list[list] = []

    cursor = start_ms
    while cursor <= end_ms:
        resp = requests.get(
            BINANCE_VISION_KLINES,
            params={
                "symbol": pair,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        if len(batch) < 1000 or last_open_ms <= cursor:
            break
        cursor = last_open_ms + 86_400_000
        time.sleep(0.05)

    if not rows:
        raise ValueError(f"No Binance klines returned for '{pair}'.")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = (
        df[["date", "open", "high", "low", "close", "volume"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df["date"] < _utc_today_naive()].reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_xaut_stitched_spot(refresh: bool = False) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Stitch XAUT-USD proxy before Binance XAUTUSDT listing with real Binance data after."""
    bn = fetch_binance_daily("XAUT/USDT", refresh=refresh)
    bn_start = bn["date"].min()
    yf_proxy = fetch_yfinance_daily("XAUT-USD", refresh=refresh)
    pre = yf_proxy[yf_proxy["date"] < bn_start].copy()
    stitched = (
        pd.concat([pre, bn], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return stitched, bn_start

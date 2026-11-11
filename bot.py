import ccxt
import json
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
import os
import time
import schedule
import datetime
from dotenv import load_dotenv
from strategy import add_indicators, check_signals

# 加載環境變數
load_dotenv()

# ================= 1. 配置與連線初始化 =================
# 幣安 API 連線 (預設正式盤實盤模式；如需切回測試網可在 .env 設 USE_TESTNET=true)
USE_TESTNET = os.getenv('USE_TESTNET', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
exchange_client = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
if USE_TESTNET:
    exchange_client.set_sandbox_mode(True)

# 資金分配與風控參數 (v3b: 保留 30% 現金備用，剩餘 70% 依權重五五分)
DEPLOY_RATIO = float(os.getenv('DEPLOY_RATIO', 0.7))      # 總資產投入比例 (0.7 = 留 30% 現金備用)
BTC_WEIGHT = float(os.getenv('BTC_WEIGHT', 0.5))          # BTC 在投入部位中的權重
XAUT_WEIGHT = float(os.getenv('XAUT_WEIGHT', 0.5))        # XAUT 在投入部位中的權重
MIN_ORDER_USDT = float(os.getenv('MIN_ORDER_USDT', 10.0)) # 買入最小下單金額 (USDT)
MIN_EXIT_USDT = float(os.getenv('MIN_EXIT_USDT', 5.5))    # 賣出最小下單門檻 (貼近幣安 5U 下限，避免小倉變死灰塵)
FETCH_MAX_RETRIES = int(os.getenv('FETCH_MAX_RETRIES', 3))
FETCH_RETRY_DELAY = float(os.getenv('FETCH_RETRY_DELAY', 5.0))
RETRY_UTC_TIMES = ("23:15", "23:30", "23:45")

# 交易對映射、權重與個別訊號模式
TRADING_CONFIG = {
    "BTC=F": {"symbol": "BTC/USDT", "weight": BTC_WEIGHT, "signal_mode": "ema30_macd"},
    "GC=F":  {"symbol": "XAUT/USDT", "weight": XAUT_WEIGHT, "signal_mode": "ema200_macd_both_exit"},
}

# 狀態持久化檔：記錄各期貨大腦已處理過的 K 線日期與失敗補抓隊列，避免容器重啟後在休市日誤觸發
STATE_FILE = Path(os.getenv('BOT_STATE_FILE', str(Path(__file__).resolve().parent / ".bot_state.json")))
LAST_PROCESSED_BAR_DATE = {}
PENDING_RETRIES = {}


def _load_state():
    """Load persisted bar dates and pending retries from STATE_FILE if present."""
    if not STATE_FILE.exists():
        return {"last_bar_date": {}, "pending": {}}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "last_bar_date": dict(data.get("last_bar_date") or {}),
                "pending": dict(data.get("pending") or {}),
            }
    except Exception as e:
        print(f"⚠️ 讀取狀態檔失敗 ({STATE_FILE}): {e}")
    return {"last_bar_date": {}, "pending": {}}


def _save_state():
    """Persist LAST_PROCESSED_BAR_DATE and PENDING_RETRIES atomically."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_bar_date": dict(LAST_PROCESSED_BAR_DATE),
            "pending": dict(PENDING_RETRIES),
        }
        tmp_path = STATE_FILE.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(STATE_FILE)
    except Exception as e:
        print(f"⚠️ 寫入狀態檔失敗 ({STATE_FILE}): {e}")


def _init_persisted_state():
    state = _load_state()
    LAST_PROCESSED_BAR_DATE.clear()
    LAST_PROCESSED_BAR_DATE.update(state["last_bar_date"])
    PENDING_RETRIES.clear()
    PENDING_RETRIES.update(state["pending"])


_init_persisted_state()


# Telegram 配置
def send_telegram_notification(msg):
    token = os.getenv('TG_BOT_TOKEN')
    chat_id = os.getenv('TG_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        # 使用 HTML 格式讓訊息更美觀
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram 推送失敗: {e}")


# ================= 2. 核心功能函數 =================

def fetch_market_data(yahoo_ticker, now_utc=None):
    """獲取期貨大腦數據 (CME/COMEX)，並剔除尚未收盤的未來日期 K 線。"""
    ticker = yf.Ticker(yahoo_ticker)
    df = ticker.history(period="max", interval="1d")
    df.reset_index(inplace=True)
    df.rename(
        columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'},
        inplace=True,
    )
    if not df.empty:
        now_utc_date = (now_utc or datetime.datetime.now(datetime.UTC)).date()
        last_candle_date = pd.Timestamp(df['timestamp'].iloc[-1]).date()
        if last_candle_date > now_utc_date:
            df = df.iloc[:-1]
    return add_indicators(df)


def fetch_market_data_with_retry(yahoo_ticker, max_retries=None, retry_delay=None):
    """多次重試抓取期貨數據，回傳 (DataFrame, attempts)。"""
    attempts_limit = max(1, FETCH_MAX_RETRIES if max_retries is None else max_retries)
    delay_sec = FETCH_RETRY_DELAY if retry_delay is None else retry_delay
    last_exc = None
    for attempt in range(1, attempts_limit + 1):
        try:
            df = fetch_market_data(yahoo_ticker)
            if df is None or len(df) < 2:
                raise ValueError(f"{yahoo_ticker} 返回有效 K 線不足 ({0 if df is None else len(df)} 根)")
            return df, attempt
        except Exception as e:
            last_exc = e
            print(f"⚠️ {yahoo_ticker} 數據抓取失敗 (第 {attempt}/{attempts_limit} 次): {e}")
            if attempt < attempts_limit and delay_sec > 0:
                time.sleep(delay_sec)
    raise last_exc


def compute_portfolio_snapshot(exchange, trading_config):
    """讀取幣安現貨餘額與即時市價，計算真實帳戶總淨值 (USDT + 持倉市值)。"""
    balance = exchange.fetch_balance()
    usdt_free = float(balance.get('free', {}).get('USDT', 0.0) or 0.0)
    usdt_total = float(balance.get('total', {}).get('USDT', 0.0) or 0.0)

    holdings = {}
    holdings_value_usdt = 0.0
    for cfg in trading_config.values():
        symbol = cfg['symbol']
        coin = symbol.split('/')[0]
        coin_free = float(balance.get('free', {}).get(coin, 0.0) or 0.0)
        coin_total = float(balance.get('total', {}).get(coin, 0.0) or 0.0)
        ticker = exchange.fetch_ticker(symbol)
        spot_price = float(ticker['last'])
        val_usdt = coin_total * spot_price
        holdings_value_usdt += val_usdt
        holdings[symbol] = {
            'coin': coin,
            'free': coin_free,
            'total': coin_total,
            'spot_price': spot_price,
            'value_usdt': val_usdt,
            'is_open': val_usdt >= MIN_ORDER_USDT,
        }

    return {
        'usdt_free': usdt_free,
        'usdt_total': usdt_total,
        'holdings': holdings,
        'total_equity_usdt': usdt_total + holdings_value_usdt,
    }


def _compute_snapshot_with_retry(exchange, trading_config, max_retries=None, retry_delay=None):
    attempts_limit = max(1, FETCH_MAX_RETRIES if max_retries is None else max_retries)
    delay_sec = FETCH_RETRY_DELAY if retry_delay is None else retry_delay
    last_exc = None
    for attempt in range(1, attempts_limit + 1):
        try:
            return compute_portfolio_snapshot(exchange, trading_config)
        except Exception as e:
            last_exc = e
            print(f"⚠️ 獲取幣安帳戶餘額失敗 (第 {attempt}/{attempts_limit} 次): {e}")
            if attempt < attempts_limit and delay_sec > 0:
                time.sleep(delay_sec)
    raise last_exc


def _is_order_executed(order):
    """確認委託確實已成交而非取消/過期/零成交。"""
    if not order:
        return False
    status = str(order.get('status') or '').lower()
    if status in {'canceled', 'cancelled', 'expired', 'rejected'}:
        return False
    filled = order.get('filled')
    if filled is not None and float(filled) <= 0:
        return False
    return True


def execute_trade(signal, symbol, target_usdt, exchange=None, coin_free=0.0):
    """執行幣安現貨交易 (市價吃單)"""
    ex = exchange or exchange_client
    try:
        if signal == "LONG_ENTRY" and target_usdt >= MIN_ORDER_USDT:
            print(f"🚀 買入執行: {symbol} | 預算: {target_usdt:.2f} USDT")
            order = ex.create_market_buy_order(symbol, target_usdt, params={'quoteOrderQty': target_usdt})
            return order if _is_order_executed(order) else None

        if signal == "LONG_EXIT" and coin_free > 0:
            print(f"📉 賣出執行: {symbol} | 數量: {coin_free}")
            order = ex.create_market_sell_order(symbol, coin_free)
            return order if _is_order_executed(order) else None
    except Exception as e:
        print(f"交易失敗 ({symbol}): {e}")
    return None


def _prune_stale_pending(today_str):
    stale_keys = [k for k, info in PENDING_RETRIES.items() if info.get("date") != today_str]
    for k in stale_keys:
        PENDING_RETRIES.pop(k, None)


def _evaluate_tickers(tickers, today_str):
    """抓取指定期貨大腦並產出訊號評估，同時標記待重試的失敗或延遲標的。"""
    evaluations = []
    for ticker in tickers:
        config = TRADING_CONFIG[ticker]
        try:
            df, attempts = fetch_market_data_with_retry(ticker)
            current = df.iloc[-1]
            bar_date = str(pd.Timestamp(current['timestamp']).date())
            already_done = LAST_PROCESSED_BAR_DATE.get(ticker) == bar_date
            behind_today = bar_date < today_str
            is_stale = already_done or behind_today
            signal = "HOLD" if is_stale else check_signals(df, signal_mode=config['signal_mode'])

            if behind_today:
                # 可能是 Yahoo 延遲或美股休市；若尚未標記 error，暫記為 lag 等待後續窗口確認
                if PENDING_RETRIES.get(ticker, {}).get("date") != today_str or PENDING_RETRIES.get(ticker, {}).get("reason") != "error":
                    PENDING_RETRIES[ticker] = {"date": today_str, "reason": "lag"}
            else:
                PENDING_RETRIES.pop(ticker, None)

            evaluations.append({
                'ticker': ticker,
                'config': config,
                'current': current,
                'bar_date': bar_date,
                'behind_today': behind_today,
                'is_stale': is_stale,
                'signal': signal,
                'attempts': attempts,
                'error': None,
            })
        except Exception as e:
            PENDING_RETRIES[ticker] = {"date": today_str, "reason": "error"}
            evaluations.append({'ticker': ticker, 'config': config, 'error': str(e), 'attempts': FETCH_MAX_RETRIES})
    return evaluations


def _execute_evaluations(evaluations, ex, now_utc, report_title="📅 <b>每日策略報告 (v3b)</b>", send_report=True):
    """依訊號先平倉再買入，完成後才提交已處理 K 線日期，並發送 Telegram 報告。"""
    today_str = str(now_utc.date())
    try:
        snapshot = _compute_snapshot_with_retry(ex, TRADING_CONFIG)
    except Exception as e:
        print(f"獲取餘額失敗: {e}")
        for ev in evaluations:
            PENDING_RETRIES[ev['ticker']] = {"date": today_str, "reason": "error"}
        if send_report:
            send_telegram_notification(
                f"❌ <b>策略執行暫停 ({now_utc.strftime('%Y-%m-%d %H:%M')} UTC)</b>\n"
                f"獲取幣安帳戶餘額失敗：<code>{e}</code>\n將於後續重試窗口自動重試，成功後補發通知。"
            )
        return False

    # 1. 先執行所有出場 (LONG_EXIT)，釋出現金後再評估進場 (LONG_ENTRY)
    trade_outcomes = {}
    failed_tickers = set()
    exit_executed = False
    for ev in evaluations:
        if ev.get('error') or ev['signal'] != "LONG_EXIT":
            continue
        ticker = ev['ticker']
        symbol = ev['config']['symbol']
        holding = snapshot['holdings'][symbol]
        free_val_usdt = holding['free'] * holding['spot_price']
        total_val_usdt = holding['value_usdt']
        if free_val_usdt >= MIN_EXIT_USDT:
            order = execute_trade("LONG_EXIT", symbol, 0.0, exchange=ex, coin_free=holding['free'])
            if order:
                trade_outcomes[symbol] = "🔴 <b>賣出成功</b> | 已清空倉位"
                exit_executed = True
            else:
                trade_outcomes[symbol] = "⚠️ <b>賣出失敗或未成交</b> (已排入重試) | 請檢查日誌"
                failed_tickers.add(ticker)
        elif total_val_usdt >= MIN_EXIT_USDT:
            trade_outcomes[symbol] = (
                f"🔒 <b>無法市價賣出</b> | 持倉已被掛單鎖定 (可用: <code>{free_val_usdt:,.2f} U</code> / "
                f"總計: <code>{total_val_usdt:,.2f} U</code>)"
            )
        else:
            trade_outcomes[symbol] = "😴 當前狀態: 空倉觀望"

    if exit_executed:
        snapshot = _compute_snapshot_with_retry(ex, TRADING_CONFIG)

    total_equity = snapshot['total_equity_usdt']
    total_strategy_budget = total_equity * DEPLOY_RATIO
    usdt_free = snapshot['usdt_free']

    daily_report = f"{report_title}\n時間: <code>{now_utc.strftime('%Y-%m-%d %H:%M')} UTC</code>\n"
    daily_report += f"帳戶總淨值: <code>{total_equity:,.2f} USDT</code>（現金: <code>{snapshot['usdt_total']:,.2f}</code>）\n"
    daily_report += f"投入目標額 ({DEPLOY_RATIO*100:.0f}%): <code>{total_strategy_budget:,.2f} USDT</code>\n\n"

    for ev in evaluations:
        symbol = ev['config']['symbol']
        ticker = ev['ticker']
        if ev.get('error'):
            daily_report += (
                f"❌ <b>{symbol} 抓取失敗 (已排入自動重試):</b> <code>{ev['error']}</code>\n\n"
            )
            continue

        current = ev['current']
        holding = snapshot['holdings'][symbol]
        target_buy_usdt = total_strategy_budget * ev['config']['weight']
        trade_status = trade_outcomes.get(symbol, "😴 當前狀態: 觀望")

        if ev['is_stale']:
            if ev.get('behind_today'):
                trade_status = f"⏸️ 期貨尚未開出今日收盤/休市 (沿用 {ev['bar_date']} K線)，略過下單"
            else:
                trade_status = f"⏸️ 期貨休市 (沿用 {ev['bar_date']} K線)，略過重複下單"
        elif ev['signal'] == "LONG_ENTRY":
            # 防止微額殘倉因價格波動略高於 10U 而吞沒整筆 35% 買入訊號
            effective_open_floor = max(MIN_ORDER_USDT, target_buy_usdt * 0.15)
            if holding['value_usdt'] >= effective_open_floor:
                trade_status = f"📌 已有持倉 (市值 {holding['value_usdt']:,.2f} U)，維持續抱"
            elif holding['total'] > holding['free'] and holding['value_usdt'] >= MIN_ORDER_USDT:
                trade_status = f"🔒 資產掛單鎖定中 (市值 {holding['value_usdt']:,.2f} U)，略過加碼"
            else:
                needed_usdt = max(0.0, target_buy_usdt - holding['value_usdt'])
                alloc_usdt = min(needed_usdt, usdt_free * 0.995)
                if alloc_usdt >= MIN_ORDER_USDT:
                    order = execute_trade("LONG_ENTRY", symbol, alloc_usdt, exchange=ex)
                    if order:
                        usdt_free -= alloc_usdt
                        trade_status = f"🚀 <b>買入成功</b> | 投入 <code>{alloc_usdt:,.2f} U</code> | 均價: <code>{order.get('average', 'Market')}</code>"
                    else:
                        trade_status = "⚠️ <b>買入失敗或未成交</b> (已排入重試) | 請檢查日誌"
                        failed_tickers.add(ticker)
                else:
                    trade_status = f"⚠️ 可用現金不足 (<code>{usdt_free:,.2f} U</code>)，略過買入"

        # 只有未出現下單失敗且非落後舊 K 線，才鎖定 bar_date
        if ticker in failed_tickers:
            PENDING_RETRIES[ticker] = {"date": today_str, "reason": "error"}
        elif not ev.get('behind_today'):
            LAST_PROCESSED_BAR_DATE[ticker] = ev['bar_date']

        retry_note = f" | 🔄 重試第 {ev['attempts']} 次成功" if ev.get('attempts', 1) > 1 else ""
        price = current['close']
        ema30 = current['ema30']
        ema200 = current['ema200']
        macd_hist = current['hist']

        report_segment = f"🔹 <b>{symbol}</b> (大腦: {ticker} | 模式: <code>{ev['config']['signal_mode']}</code>{retry_note})\n"
        report_segment += f"• K線日期: <code>{ev['bar_date']}</code> | 期貨價: <code>{price:,.2f}</code> | Hist: <code>{macd_hist:.2f}</code>\n"
        report_segment += f"• 均線指標: EMA30: <code>{ema30:,.2f}</code> | EMA200: <code>{ema200:,.2f}</code>\n"
        report_segment += f"• 目標倉位: <code>{target_buy_usdt:,.2f} USDT</code> | 現持市值: <code>{holding['value_usdt']:,.2f} USDT</code>\n"
        report_segment += f"• <b>{trade_status}</b>\n\n"
        daily_report += report_segment

    if send_report:
        send_telegram_notification(daily_report)
        print("✅ 策略報告已推送至 Telegram。")
    return len(failed_tickers) == 0


# ================= 3. 策略執行排程 =================

def run_strategy(exchange=None, now_utc=None):
    ex = exchange or exchange_client
    now_utc = now_utc or datetime.datetime.now(datetime.UTC)

    # 週末濾網
    if now_utc.weekday() in [5, 6]:
        print(f"⏰ [{now_utc.strftime('%Y-%m-%d')}] 週末休市中，略過判斷。")
        return

    today_str = str(now_utc.date())
    _prune_stale_pending(today_str)

    evaluations = _evaluate_tickers(list(TRADING_CONFIG.keys()), today_str)
    _execute_evaluations(evaluations, ex, now_utc, report_title="📅 <b>每日策略報告 (v3b)</b>", send_report=True)
    _save_state()


def retry_pending_strategy(exchange=None, now_utc=None, is_final_attempt=False):
    """針對稍早抓取失敗或尚未開出當日 K 線的標的進行自動補抓，成功時發送 Telegram 通知。"""
    ex = exchange or exchange_client
    now_utc = now_utc or datetime.datetime.now(datetime.UTC)

    if now_utc.weekday() in [5, 6]:
        return

    today_str = str(now_utc.date())
    _prune_stale_pending(today_str)

    pending_before = {
        ticker: dict(info)
        for ticker, info in PENDING_RETRIES.items()
        if info.get("date") == today_str and ticker in TRADING_CONFIG
    }
    if not pending_before:
        return

    evaluations = _evaluate_tickers(list(pending_before.keys()), today_str)

    recovered_evs = []
    still_errors = []
    for ev in evaluations:
        ticker = ev['ticker']
        prev_reason = pending_before[ticker].get("reason")
        if ev.get('error'):
            still_errors.append(ev)
            continue
        if prev_reason == "error":
            # 先前曾報錯中斷抓取，現在終於抓成功 -> 清除 error 狀態並發送通知
            if ev.get('behind_today'):
                PENDING_RETRIES[ticker] = {"date": today_str, "reason": "lag"}
            recovered_evs.append(ev)
        elif prev_reason == "lag" and not ev['is_stale']:
            # 先前 K 線延遲尚未出來，現在終於等到今日收盤新柱
            recovered_evs.append(ev)

    if recovered_evs:
        _execute_evaluations(
            recovered_evs,
            ex,
            now_utc,
            report_title="🔄 <b>數據補抓成功並完成執行通知 (v3b)</b>",
            send_report=True,
        )

    if is_final_attempt:
        if still_errors:
            err_lines = "\n".join(
                f"• <code>{ev['config']['symbol']}</code> ({ev['ticker']}): <code>{ev['error']}</code>"
                for ev in still_errors
            )
            send_telegram_notification(
                f"❌ <b>今日數據補抓多次重試仍失敗 ({now_utc.strftime('%Y-%m-%d %H:%M')} UTC)</b>\n"
                f"{err_lines}\n請檢查網路連線或 Yahoo Finance 數據源。"
            )
        # 今日最後一個重試窗結束，清空待重試標記避免明日殘留
        for ticker in pending_before:
            PENDING_RETRIES.pop(ticker, None)

    _save_state()


# ================= 4. 主程式啟動 =================

if __name__ == "__main__":
    env_mode = "🧪 測試網 (Testnet)" if USE_TESTNET else "🔴 實盤 (Mainnet)"
    start_msg = f"🤖 <b>CME/COMEX 雙腦機器人已啟動 (v3b | {env_mode})</b>\n"
    start_msg += "監控標的: BTC (EMA30+EMA200+MACD) & XAUT (EMA200+MACD 雙重確認)\n"
    start_msg += f"資金配置: 保留 {(1-DEPLOY_RATIO)*100:.0f}% 備用 | {DEPLOY_RATIO*100:.0f}% 按 {BTC_WEIGHT*100:.0f}%/{XAUT_WEIGHT*100:.0f}% 分配\n"
    start_msg += f"掃描時間: 每日 23:00 UTC (失敗/延遲自動重試: {', '.join(RETRY_UTC_TIMES)} UTC)"

    print(f"🤖 機器人進入循環監控模式 ({env_mode})...")
    send_telegram_notification(start_msg)

    # 每天 23:00 UTC 執行 (CME 收盤時間，明確鎖定 UTC)
    schedule.every().day.at("23:00", "UTC").do(run_strategy)
    for idx, retry_time in enumerate(RETRY_UTC_TIMES):
        is_last = idx == len(RETRY_UTC_TIMES) - 1
        schedule.every().day.at(retry_time, "UTC").do(
            retry_pending_strategy, is_final_attempt=is_last
        )

    while True:
        schedule.run_pending()
        # 每小時在 console 打印一次心跳，確保 Docker logs 有東西看
        now = datetime.datetime.now(datetime.UTC)
        if now.minute == 0 and now.second == 0:
            print(f"💓 心跳檢查: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC - 運作正常")
            time.sleep(1)  # 防止一秒內重複打印
        time.sleep(1)

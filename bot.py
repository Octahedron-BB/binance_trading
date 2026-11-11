import ccxt
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
# 幣安 API 連線
exchange_testnet = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
# 注意：實盤請務必將此行註解掉或設為 False
exchange_testnet.set_sandbox_mode(True)

# 資金分配參數 (v3b: 保留 30% 現金備用，剩餘 70% 依權重五五分)
DEPLOY_RATIO = float(os.getenv('DEPLOY_RATIO', 0.7))      # 總資產投入比例 (0.7 = 留 30% 現金備用)
BTC_WEIGHT = float(os.getenv('BTC_WEIGHT', 0.5))          # BTC 在投入部位中的權重
XAUT_WEIGHT = float(os.getenv('XAUT_WEIGHT', 0.5))        # XAUT 在投入部位中的權重
MIN_ORDER_USDT = float(os.getenv('MIN_ORDER_USDT', 10.0)) # 低於此淨值視為零頭/空倉

# 交易對映射、權重與個別訊號模式
TRADING_CONFIG = {
    "BTC=F": {"symbol": "BTC/USDT", "weight": BTC_WEIGHT, "signal_mode": "ema30_macd"},
    "GC=F":  {"symbol": "XAUT/USDT", "weight": XAUT_WEIGHT, "signal_mode": "ema200_macd_both_exit"},
}

# 記錄各期貨大腦已處理過的最新 K 線日期，防止美股休市日重複觸發同一根舊 K 線
LAST_PROCESSED_BAR_DATE = {}


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


def execute_trade(signal, symbol, target_usdt, exchange=None, coin_free=0.0):
    """執行幣安現貨交易 (市價吃單)"""
    ex = exchange or exchange_testnet
    try:
        if signal == "LONG_ENTRY" and target_usdt >= MIN_ORDER_USDT:
            print(f"🚀 買入執行: {symbol} | 預算: {target_usdt:.2f} USDT")
            order = ex.create_market_buy_order(symbol, target_usdt, params={'quoteOrderQty': target_usdt})
            return order

        if signal == "LONG_EXIT" and coin_free > 0:
            print(f"📉 賣出執行: {symbol} | 數量: {coin_free}")
            order = ex.create_market_sell_order(symbol, coin_free)
            return order
    except Exception as e:
        print(f"交易失敗 ({symbol}): {e}")
    return None


# ================= 3. 策略執行排程 =================

def run_strategy(exchange=None, now_utc=None):
    ex = exchange or exchange_testnet
    now_utc = now_utc or datetime.datetime.now(datetime.UTC)

    # 週末濾網
    if now_utc.weekday() in [5, 6]:
        print(f"⏰ [{now_utc.strftime('%Y-%m-%d')}] 週末休市中，略過判斷。")
        return

    try:
        snapshot = compute_portfolio_snapshot(ex, TRADING_CONFIG)
    except Exception as e:
        print(f"獲取餘額失敗: {e}")
        return

    # 先讀取全部標的大腦訊號並過濾美股假期重複舊 K 線
    evaluations = []
    for ticker, config in TRADING_CONFIG.items():
        symbol = config['symbol']
        try:
            df = fetch_market_data(ticker)
            current = df.iloc[-1]
            bar_date = str(pd.Timestamp(current['timestamp']).date())
            is_stale = LAST_PROCESSED_BAR_DATE.get(ticker) == bar_date
            signal = "HOLD" if is_stale else check_signals(df, signal_mode=config['signal_mode'])
            LAST_PROCESSED_BAR_DATE[ticker] = bar_date
            evaluations.append({
                'ticker': ticker,
                'config': config,
                'current': current,
                'bar_date': bar_date,
                'is_stale': is_stale,
                'signal': signal,
                'error': None,
            })
        except Exception as e:
            evaluations.append({'ticker': ticker, 'config': config, 'error': str(e)})

    # 先執行所有出場 (LONG_EXIT)，釋出現金後再評估進場 (LONG_ENTRY)
    trade_outcomes = {}
    exit_executed = False
    for ev in evaluations:
        if ev.get('error') or ev['signal'] != "LONG_EXIT":
            continue
        symbol = ev['config']['symbol']
        holding = snapshot['holdings'][symbol]
        if holding['free'] * holding['spot_price'] >= MIN_ORDER_USDT:
            order = execute_trade("LONG_EXIT", symbol, 0.0, exchange=ex, coin_free=holding['free'])
            if order:
                trade_outcomes[symbol] = "🔴 <b>賣出成功</b> | 已清空倉位"
                exit_executed = True
            else:
                trade_outcomes[symbol] = "⚠️ <b>賣出失敗</b> | 請檢查日誌"
        else:
            trade_outcomes[symbol] = "😴 當前狀態: 空倉觀望"

    if exit_executed:
        snapshot = compute_portfolio_snapshot(ex, TRADING_CONFIG)

    total_equity = snapshot['total_equity_usdt']
    total_strategy_budget = total_equity * DEPLOY_RATIO
    usdt_free = snapshot['usdt_free']

    daily_report = f"📅 <b>每日策略報告 (v3b)</b>\n時間: <code>{now_utc.strftime('%Y-%m-%d %H:%M')} UTC</code>\n"
    daily_report += f"帳戶總淨值: <code>{total_equity:,.2f} USDT</code>（現金: <code>{snapshot['usdt_total']:,.2f}</code>）\n"
    daily_report += f"投入目標額 ({DEPLOY_RATIO*100:.0f}%): <code>{total_strategy_budget:,.2f} USDT</code>\n\n"

    for ev in evaluations:
        symbol = ev['config']['symbol']
        ticker = ev['ticker']
        if ev.get('error'):
            daily_report += f"❌ <b>{symbol} 運行出錯:</b> <code>{ev['error']}</code>\n\n"
            continue

        current = ev['current']
        holding = snapshot['holdings'][symbol]
        target_buy_usdt = total_strategy_budget * ev['config']['weight']
        trade_status = trade_outcomes.get(symbol, "😴 當前狀態: 觀望")

        if ev['is_stale']:
            trade_status = f"⏸️ 期貨休市 (沿用 {ev['bar_date']} K線)，略過重複下單"
        elif ev['signal'] == "LONG_ENTRY":
            if holding['is_open']:
                trade_status = f"📌 已有持倉 (市值 {holding['value_usdt']:,.2f} U)，維持續抱"
            else:
                alloc_usdt = min(target_buy_usdt, usdt_free * 0.995)
                if alloc_usdt >= MIN_ORDER_USDT:
                    order = execute_trade("LONG_ENTRY", symbol, alloc_usdt, exchange=ex)
                    if order:
                        usdt_free -= alloc_usdt
                        trade_status = f"🚀 <b>買入成功</b> | 投入 <code>{alloc_usdt:,.2f} U</code> | 均價: <code>{order.get('average', 'Market')}</code>"
                    else:
                        trade_status = "⚠️ <b>買入失敗</b> | 請檢查日誌"
                else:
                    trade_status = f"⚠️ 可用現金不足 (<code>{usdt_free:,.2f} U</code>)，略過買入"

        price = current['close']
        ema30 = current['ema30']
        ema200 = current['ema200']
        macd_hist = current['hist']

        report_segment = f"🔹 <b>{symbol}</b> (大腦: {ticker} | 模式: <code>{ev['config']['signal_mode']}</code>)\n"
        report_segment += f"• 期貨價: <code>{price:,.2f}</code> | Hist: <code>{macd_hist:.2f}</code>\n"
        report_segment += f"• 均線指標: EMA30: <code>{ema30:,.2f}</code> | EMA200: <code>{ema200:,.2f}</code>\n"
        report_segment += f"• 目標倉位: <code>{target_buy_usdt:,.2f} USDT</code> | 現持市值: <code>{holding['value_usdt']:,.2f} USDT</code>\n"
        report_segment += f"• <b>{trade_status}</b>\n\n"
        daily_report += report_segment

    send_telegram_notification(daily_report)
    print("✅ 每日任務執行完畢，報告已推送。")


# ================= 4. 主程式啟動 =================

if __name__ == "__main__":
    start_msg = "🤖 <b>CME/COMEX 雙腦機器人已啟動 (v3b)</b>\n"
    start_msg += "監控標的: BTC (EMA30+EMA200+MACD) & XAUT (EMA200+MACD 雙重確認)\n"
    start_msg += f"資金配置: 保留 {(1-DEPLOY_RATIO)*100:.0f}% 備用 | {DEPLOY_RATIO*100:.0f}% 按 {BTC_WEIGHT*100:.0f}%/{XAUT_WEIGHT*100:.0f}% 分配\n"
    start_msg += "掃描時間: 每日 23:00 UTC (鎖定 UTC 時區)"

    print("🤖 機器人進入循環監控模式...")
    send_telegram_notification(start_msg)

    # 每天 23:00 UTC 執行 (CME 收盤時間，明確鎖定 UTC)
    schedule.every().day.at("23:00", "UTC").do(run_strategy)

    while True:
        schedule.run_pending()
        # 每小時在 console 打印一次心跳，確保 Docker logs 有東西看
        now = datetime.datetime.now(datetime.UTC)
        if now.minute == 0 and now.second == 0:
            print(f"💓 心跳檢查: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC - 運作正常")
            time.sleep(1)  # 防止一秒內重複打印
        time.sleep(1)

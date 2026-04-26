import ccxt
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import os
import time
import schedule
import datetime
from dotenv import load_dotenv

load_dotenv()

# ================= 連線與參數初始化 =================
exchange_testnet = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
exchange_testnet.set_sandbox_mode(True) 

PORTFOLIO_RATIO = float(os.getenv('PORTFOLIO_RATIO', 0.5))
INVESTMENT_RATIO = float(os.getenv('INVESTMENT_RATIO', 0.7))

def send_telegram_notification(msg):
    token = os.getenv('TG_BOT_TOKEN')
    chat_id = os.getenv('TG_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

# ================= 數據獲取 (Yahoo Finance 通用版) =================
def fetch_market_data(yahoo_ticker):
    ticker = yf.Ticker(yahoo_ticker)
    df = ticker.history(period="1y", interval="1d")
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    df.ta.ema(length=30, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.rename(columns={'EMA_30':'ema30', 'EMA_200':'ema200', 'MACDh_12_26_9':'hist'}, inplace=True)
    df.dropna(subset=['ema200', 'hist'], inplace=True)
    return df

# ================= 核心邏輯判斷 =================
def check_signals(df):
    current = df.iloc[-1]
    prev = df.iloc[-2]
    macro_bull = current['close'] > current['ema200']
    
    cond_a = (prev['hist'] <= 0 and current['hist'] > 0) and (current['close'] > current['ema30'])
    cond_b = (prev['close'] <= prev['ema30'] and current['close'] > current['ema30']) and (current['hist'] > 0)
    
    if current['hist'] < 0 or current['close'] < current['ema30']:
        return "LONG_EXIT"
    if (cond_a or cond_b) and macro_bull:
        return "LONG_ENTRY"
    return "HOLD"

# ================= 交易執行 =================
def execute_trade(signal, symbol):
    try:
        balance = exchange_testnet.fetch_balance()
        coin_name = symbol.split('/')[0]
        
        if signal == "LONG_ENTRY":
            usdt_total = balance['total'].get('USDT', 0)
            buy_amount_usdt = usdt_total * PORTFOLIO_RATIO * INVESTMENT_RATIO
            order = exchange_testnet.create_market_buy_order(symbol, buy_amount_usdt, params={'quoteOrderQty': buy_amount_usdt})
            return order

        elif signal == "LONG_EXIT":
            coin_free = balance['free'].get(coin_name, 0)
            if coin_free > 0:
                order = exchange_testnet.create_market_sell_order(symbol, coin_free)
                return order
    except Exception as e:
        print(f"❌ 交易執行失敗: {e}")
    return None

# ================= 每日掃描任務 =================
def run_strategy():
    now_utc = datetime.datetime.now(datetime.UTC)
    if now_utc.weekday() in [5, 6]:
        print(f"⏰ [{now_utc.strftime('%Y-%m-%d')}] 週末期貨休市。")
        return

    trading_pairs = {"BTC=F": "BTC/USDT", "GC=F": "PAXG/USDT"}
    daily_report = f"📅 <b>每日交易狀態報告</b>\n時間: <code>{now_utc.strftime('%Y-%m-%d %H:%M')} UTC</code>\n\n"

    for yahoo_ticker, binance_symbol in trading_pairs.items():
        try:
            df = fetch_market_data(yahoo_ticker)
            current = df.iloc[-1]
            signal = check_signals(df)
            
            report_segment = f"🔹 <b>標的: {binance_symbol}</b>\n"
            report_segment += f"• 期貨價: <code>{current['close']:,.2f}</code> | Hist: <code>{current['hist']:.2f}</code>\n"
            report_segment += f"• 宏觀趨勢: {"📈 多頭" if current['close'] > current['ema200'] else "📉 空頭"}\n"

            trade_info = "😴 <b>當前狀態:</b> 持續觀望"
            if signal == "LONG_ENTRY":
                order = execute_trade(signal, binance_symbol)
                if order:
                    trade_info = f"🚀 <b>【買入執行】</b> 價位: <code>{order.get('average', 'Market')}</code>"
            elif signal == "LONG_EXIT":
                order = execute_trade(signal, binance_symbol)
                if order: trade_info = "🔴 <b>【賣出執行】</b> 已全倉清空。"

            report_segment += f"{trade_info}\n\n"
            daily_report += report_segment
        except Exception as e:
            daily_report += f"🔹 <b>{binance_symbol}</b>\n❌ 錯誤: {str(e)}\n\n"

    send_telegram_notification(daily_report)

if __name__ == "__main__":
    print("🤖 CME/COMEX 雙腦機器人已啟動...")
    schedule.every().day.at("23:00").do(run_strategy)
    while True:
        schedule.run_pending()
        time.sleep(60)
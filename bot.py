import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import os
import time
import schedule
import datetime
from dotenv import load_dotenv

load_dotenv()

# ================= 連線與參數初始化 =================

# 1. 執行專用 (Testnet)
exchange_testnet = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
exchange_testnet.set_sandbox_mode(True) 

# 2. 數據專用 (Mainnet)
exchange_mainnet = ccxt.binance({'enableRateLimit': True})

# 3. 策略常數
PORTFOLIO_RATIO = float(os.getenv('PORTFOLIO_RATIO', 0.5))
INVESTMENT_RATIO = float(os.getenv('INVESTMENT_RATIO', 0.7))

# ================= 核心組件 =================

def send_telegram_notification(msg):
    token = os.getenv('TG_BOT_TOKEN')
    chat_id = os.getenv('TG_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

def fetch_and_analyze(symbol, timeframe='1d', limit=250):
    bars = exchange_mainnet.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.ta.ema(length=30, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.rename(columns={'EMA_30':'ema30', 'EMA_200':'ema200', 'MACDh_12_26_9':'hist'}, inplace=True)
    df.dropna(subset=['ema200', 'hist'], inplace=True)
    return df

def check_signals(df, symbol):
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 週末濾網判定 (僅 PAXG 適用)
    now_utc = datetime.datetime.now(datetime.UTC)
    is_weekend = now_utc.weekday() in [5, 6]
    
    macro_bull = current['close'] > current['ema200']
    cond_a = (prev['hist'] <= 0 and current['hist'] > 0) and (current['close'] > current['ema30'])
    cond_b = (prev['close'] <= prev['ema30'] and current['close'] > current['ema30']) and (current['hist'] > 0)
    
    # 出場邏輯
    if current['hist'] < 0 or current['close'] < current['ema30']:
        return "LONG_EXIT"
        
    # 進場邏輯 (含週末濾網)
    if (cond_a or cond_b) and macro_bull:
        if symbol.startswith("PAXG") and is_weekend:
            print(f"⏳ {symbol} 觸發進場訊號，但因週末濾網略過。")
            return "HOLD"
        return "LONG_ENTRY"
        
    return "HOLD"

def execute_trade(signal, symbol):
    try:
        balance = exchange_testnet.fetch_balance()
        coin_name = symbol.split('/')[0]
        
        if signal == "LONG_ENTRY":
            # 以當前 USDT 總餘額計算下單量
            usdt_total = balance['total'].get('USDT', 0)
            buy_amount_usdt = usdt_total * PORTFOLIO_RATIO * INVESTMENT_RATIO
            print(f"🚀 執行買入 {symbol}: 約 {buy_amount_usdt} USDT")
            order = exchange_testnet.create_market_buy_order(symbol, buy_amount_usdt, params={'quoteOrderQty': buy_amount_usdt})
            return order

        elif signal == "LONG_EXIT":
            coin_free = balance['free'].get(coin_name, 0)
            if coin_free > 0:
                print(f"📉 執行賣出 {symbol}: 全倉市價賣出 {coin_free}")
                order = exchange_testnet.create_market_sell_order(symbol, coin_free)
                return order
    except Exception as e:
        print(f"❌ 交易執行失敗: {e}")
    return None

# ================= 任務分派 =================

def run_strategy():
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n⏰ [{now} UTC] 定時任務啟動：開始掃描標的...")
    
    targets = ['BTC/USDT', 'PAXG/USDT']
    
    for symbol in targets:
        try:
            df = fetch_and_analyze(symbol)
            signal = check_signals(df, symbol)
            price = df.iloc[-1]['close']
            
            print(f" > {symbol} 判斷結果: {signal}")
            
            if signal == "LONG_ENTRY":
                if execute_trade(signal, symbol):
                    send_telegram_notification(f"🟢 <b>【已執行買入】</b>\n標的: {symbol}\n價格: {price:.2f}")
            elif signal == "LONG_EXIT":
                if execute_trade(signal, symbol):
                    send_telegram_notification(f"🔴 <b>【已執行賣出】</b>\n標的: {symbol}\n價格: {price:.2f}")
        except Exception as e:
            print(f"❌ 處理 {symbol} 時發生錯誤: {e}")

# ================= 啟動循環 =================

if __name__ == "__main__":
    print("🤖 量化機器人已啟動，進入 24 小時守候模式...")
    send_telegram_notification("🤖 <b>量化機器人上線成功！</b>\n系統已進入 24 小時監控模式，將於每日日線收盤後自動執行。")

    # 設定每天 UTC 00:01 執行一次 (對應台北時間 08:01)
    # 大多數 VPS 的系統時間都是 UTC，所以設定 00:01 是標準做法
    schedule.every().day.at("00:01").do(run_strategy)

    # 為了方便你部署後立刻看到效果，我們可以手動先跑一次
    # run_strategy() 

    while True:
        schedule.run_pending()
        # 每分鐘印出一次心跳日誌，確保程式沒掛掉
        if datetime.datetime.now().minute == 0:
            print(f"💓 系統正常運行中... 當前時間: {datetime.datetime.now().strftime('%H:%M')}")
        time.sleep(60)
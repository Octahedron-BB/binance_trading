import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

exchange_testnet = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
exchange_testnet.set_sandbox_mode(True) 

exchange_mainnet = ccxt.binance({'enableRateLimit': True})

def execute_trade(signal, symbol, current_price):
    try:
        balance = exchange_testnet.fetch_balance()
        usdt_free = balance['free'].get('USDT', 0)
        
        coin_name = symbol.split('/')[0]
        coin_free = balance['free'].get(coin_name, 0)

        p_ratio = float(os.getenv('PORTFOLIO_RATIO', 0.5))
        i_ratio = float(os.getenv('INVESTMENT_RATIO', 0.7))

        if signal == "LONG_ENTRY":
            total_equity = balance['total'].get('USDT', 0)
            buy_amount_usdt = total_equity * p_ratio * i_ratio
            
            if usdt_free < buy_amount_usdt:
                print(f"⚠️ 餘額不足。需要: {buy_amount_usdt:.2f}, 現有: {usdt_free:.2f}")
                return None

            print(f"🚀 執行買入：以市價購買約 {buy_amount_usdt:.2f} USDT 的 {symbol}")
            order = exchange_testnet.create_market_buy_order(symbol, buy_amount_usdt, params={'quoteOrderQty': buy_amount_usdt})
            return order

        elif signal == "LONG_EXIT":
            if coin_free <= 0:
                print(f"ℹ️ 目前未持有 {coin_name}，無需執行出場。")
                return None
            
            print(f"📉 執行賣出：全倉市價賣出 {coin_free} 個 {symbol}")
            order = exchange_testnet.create_market_sell_order(symbol, coin_free)
            return order

    except Exception as e:
        print(f"❌ 交易執行失敗 ({symbol}): {e}")
        return None


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
    
    now_utc = datetime.datetime.now(datetime.UTC)
    is_weekend = now_utc.weekday() in [5, 6] 
    allow_entry = True
    
    if symbol == 'PAXG/USDT' and is_weekend:
        allow_entry = False
        print("⏳ [週末濾網生效] PAXG 週末不執行新進場。")

    macro_bull = current['close'] > current['ema200']
    cond_a = (prev['hist'] <= 0 and current['hist'] > 0) and (current['close'] > current['ema30'])
    cond_b = (prev['close'] <= prev['ema30'] and current['close'] > current['ema30']) and (current['hist'] > 0)
    
    long_condition = (cond_a or cond_b) and macro_bull and allow_entry
    exit_condition = (current['hist'] < 0) or (current['close'] < current['ema30'])
    
    if long_condition: return "LONG_ENTRY"
    if exit_condition: return "LONG_EXIT"
    return "HOLD"

def send_telegram_notification(msg):
    token = os.getenv('TG_BOT_TOKEN')
    chat_id = os.getenv('TG_CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

if __name__ == "__main__":
    try:
        balance = exchange_testnet.fetch_balance()
        print(f"✅ 測試網帳戶驗證成功！目前餘額: {balance['total'].get('USDT', 0):.2f} USDT\n")
        
        target_symbols = ['BTC/USDT', 'PAXG/USDT']
        
        for symbol in target_symbols:
            print(f"--- 開始處理 {symbol} ---")
            try:
                data = fetch_and_analyze(symbol)
                signal = check_signals(data, symbol)
                latest_price = data.iloc[-1]['close']
                
                print(f"當前訊號: {signal}")

                if signal == "LONG_ENTRY":
                    order = execute_trade(signal, symbol, latest_price)
                    if order:
                        send_telegram_notification(f"🟢 <b>已執行買入</b>\n標的: {symbol}\n均價: {latest_price:.2f}")
                elif signal == "LONG_EXIT":
                    order = execute_trade(signal, symbol, latest_price)
                    if order:
                        send_telegram_notification(f"🔴 <b>已執行賣出</b>\n標的: {symbol}\n均價: {latest_price:.2f}")
                else:
                    print("目前無訊號，系統待機中。")
                    
            except Exception as e:
                print(f"⚠️ {symbol} 處理發生錯誤 (測試網可能無此幣種): {e}")
            
            print("\n")
            
    except Exception as e:
        print(f"系統運行錯誤: {e}")
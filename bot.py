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

# 資金分配參數
PORTFOLIO_RATIO = float(os.getenv('PORTFOLIO_RATIO', 0.5))    # 拿出多少總資產跑策略
INVESTMENT_RATIO = float(os.getenv('INVESTMENT_RATIO', 0.7)) # 策略預算中，每次進場的比例
BTC_WEIGHT = float(os.getenv('BTC_WEIGHT', 0.5))             # BTC 預算權重
XAUT_WEIGHT = float(os.getenv('XAUT_WEIGHT', 0.5))           # XAUT 預算權重

# 交易對映射與權重
TRADING_CONFIG = {
    "BTC=F": {"symbol": "BTC/USDT", "weight": BTC_WEIGHT},
    "GC=F":  {"symbol": "XAUT/USDT", "weight": XAUT_WEIGHT}
}

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

def fetch_market_data(yahoo_ticker):
    """獲取期貨大腦數據 (CME/COMEX)"""
    ticker = yf.Ticker(yahoo_ticker)
    # 抓取 1 年份的日線數據
    df = ticker.history(period="max", interval="1d")
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    # 剔除未收盤數據
    now_utc_data = datetime.datetime.now(datetime.UTC).date()
    last_candle_date = df['timestamp'].iloc[-1].date()
    if last_candle_date > now_utc_data:
        df = df.iloc[:-1]
        
    # 計算指標
    df.ta.ema(length=30, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 清理列名 (pandas_ta 產生的名稱可能較長)
    df.rename(columns={'EMA_30':'ema30', 'EMA_200':'ema200', 'MACDh_12_26_9':'hist'}, inplace=True)
    df.dropna(subset=['ema200', 'hist'], inplace=True)
    return df

def check_signals(df):
    """真・原始版邏輯判斷 (僅根據期貨數據)"""
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    macro_bull = current['close'] > current['ema200']
    
    # 進場 A: MACD 金叉 + 價高於 EMA30
    cond_a = (prev['hist'] <= 0 and current['hist'] > 0) and (current['close'] > current['ema30'])
    # 進場 B: 價穿 EMA30 + MACD 柱狀 > 0
    cond_b = (prev['close'] <= prev['ema30'] and current['close'] > current['ema30']) and (current['hist'] > 0)
    
    # 出場: MACD 轉負 或 跌破 EMA30
    if current['hist'] < 0 or current['close'] < current['ema30']:
        return "LONG_EXIT"
        
    if (cond_a or cond_b) and macro_bull:
        return "LONG_ENTRY"
        
    return "HOLD"

def execute_trade(signal, symbol, target_usdt):
    """執行幣安現貨交易 (市價吃單)"""
    try:
        balance = exchange_testnet.fetch_balance()
        coin_name = symbol.split('/')[0]
        
        if signal == "LONG_ENTRY":
            print(f"🚀 買入執行: {symbol} | 預算: {target_usdt:.2f} USDT")
            order = exchange_testnet.create_market_buy_order(symbol, target_usdt, params={'quoteOrderQty': target_usdt})
            return order

        elif signal == "LONG_EXIT":
            coin_free = balance['free'].get(coin_name, 0)
            if coin_free > 0:
                print(f"📉 賣出執行: {symbol} | 數量: {coin_free}")
                order = exchange_testnet.create_market_sell_order(symbol, coin_free)
                return order
    except Exception as e:
        print(f"交易失敗 ({symbol}): {e}")
    return None

# ================= 3. 策略執行排程 =================

def run_strategy():
    now_utc = datetime.datetime.now(datetime.UTC)
    
    # 週末濾網
    if now_utc.weekday() in [5, 6]:
        print(f"⏰ [{now_utc.strftime('%Y-%m-%d')}] 週末休市中，略過判斷。")
        return

    # 計算總預算分配 (50/50 邏輯)
    try:
        balance = exchange_testnet.fetch_balance()
        total_usdt = balance['total'].get('USDT', 0)
        total_strategy_budget = total_usdt * PORTFOLIO_RATIO * INVESTMENT_RATIO
    except Exception as e:
        print(f"獲取餘額失敗: {e}")
        return

    daily_report = f"📅 <b>每日策略報告</b>\n時間: <code>{now_utc.strftime('%Y-%m-%d %H:%M')} UTC</code>\n"
    daily_report += f"總 USDT 餘額: <code>{total_usdt:,.2f}</code>\n"
    daily_report += f"今日總預算: <code>{total_strategy_budget:,.2f}</code>\n\n"

    for ticker, config in TRADING_CONFIG.items():
        symbol = config['symbol']
        weight = config['weight']
        
        try:
            df = fetch_market_data(ticker)
            signal = check_signals(df)
            current = df.iloc[-1]
            
            target_buy_usdt = total_strategy_budget * weight
            
            # 提取技術指標數值 (用於日誌)
            price = current['close']
            ema30 = current['ema30']
            ema200 = current['ema200']
            macd_hist = current['hist']

            report_segment = f"🔹 <b>{symbol}</b> (大腦: {ticker})\n"
            report_segment += f"• 期貨價: <code>{price:,.2f}</code> | Hist: <code>{macd_hist:.2f}</code>\n"
            report_segment += f"• 均線指標: EMA30: <code>{ema30:,.2f}</code> | EMA200: <code>{ema200:,.2f}</code>\n"
            report_segment += f"• 分配預算: <code>{target_buy_usdt:,.2f} USDT</code>\n"

            trade_status = "😴 當前狀態: 觀望"
            if signal == "LONG_ENTRY":
                order = execute_trade(signal, symbol, target_buy_usdt)
                if order: trade_status = f"🚀 <b>買入成功</b> | 均價: <code>{order.get('average', 'Market')}</code>"
            elif signal == "LONG_EXIT":
                order = execute_trade(signal, symbol, 0)
                if order: trade_status = "🔴 <b>賣出成功</b> | 已清空倉位"

            report_segment += f"• <b>{trade_status}</b>\n\n"
            daily_report += report_segment
            
        except Exception as e:
            daily_report += f"❌ <b>{symbol} 運行出錯:</b> <code>{str(e)}</code>\n\n"

    send_telegram_notification(daily_report)
    print("✅ 每日任務執行完畢，報告已推送。")

# ================= 4. 主程式啟動 =================

if __name__ == "__main__":
    start_msg = "🤖 <b>CME/COMEX 雙腦機器人已啟動</b>\n"
    start_msg += f"監控標的: BTC & XAUT\n"
    start_msg += f"預算權重: {BTC_WEIGHT*100}% / {XAUT_WEIGHT*100}%\n"
    start_msg += "掃描時間: 每日 23:00 UTC"
    
    print("🤖 機器人進入循環監控模式...")
    send_telegram_notification(start_msg)

    # 每天 23:00 執行 (CME 收盤時間)
    schedule.every().day.at("23:00").do(run_strategy)

    while True:
        schedule.run_pending()
        # 每小時在 console 打印一次心跳，確保 Docker logs 有東西看
        now = datetime.datetime.now()
        if now.minute == 0 and now.second == 0:
            print(f"💓 心跳檢查: {now.strftime('%Y-%m-%d %H:%M:%S')} - 運作正常")
            time.sleep(1) # 防止一秒內重複打印
        time.sleep(1)
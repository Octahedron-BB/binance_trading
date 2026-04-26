# Binance Trading Bot

這是一個基於 Python 的自動化加密貨幣交易機器人，使用 `ccxt` 串接 Binance API，並結合 `pandas_ta` 進行技術指標分析。

## 🚀 功能特點

- **多幣種掃描**：預設監控 BTC/USDT 與 PAXG/USDT。
- **技術指標策略**：結合 EMA (30/200) 與 MACD 進行趨勢判斷與進出場訊號捕捉。
- **週末濾網**：針對特定幣種 (如 PAXG) 設有週末不進場機制，規避流動性風險。
- **即時通知**：透過 Telegram Bot 發送買入/賣出執行結果。
- **安全交易**：支援 Binance Testnet 測試網進行模擬交易。
- **容器化支援**：提供 Dockerfile，方便在雲端伺服器快速佈署。

## 🛠️ 快速開始

### 1. 環境設定
在專案根目錄建立 `.env` 檔案，並填入以下內容：

```env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
TG_BOT_TOKEN=your_telegram_bot_token
TG_CHAT_ID=your_telegram_chat_id
PORTFOLIO_RATIO=0.5
INVESTMENT_RATIO=0.7
```

### 2. 本地執行
確保已安裝 Python 3.12+，然後執行：

```bash
pip install -r requirements.txt
python bot.py
```

### 3. 使用 Docker 佈署
```bash
docker build -t binance-bot .
docker run -d --env-file .env binance-bot
```

## 📈 交易策略簡介
- **進場條件 (LONG)**：
  - 價格高於 EMA 200 (長期牛市趨勢)。
  - MACD 柱狀圖由負轉正 且 價格高於 EMA 30。
- **出場條件 (EXIT)**：
  - MACD 柱狀圖轉負 或 價格跌破 EMA 30。

## ⚖️ 免責聲明
本專案僅供學習與研究使用，投資有風險，使用自動化交易程式請謹慎評估並自行承擔風險。

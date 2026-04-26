# 使用輕量級 Python 映像檔
FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 先複製依賴清單並安裝 (利用 Docker 快取機制)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY . .

# 啟動命令
CMD ["python", "bot.py"]
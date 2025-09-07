import os
import time
import requests
import numpy as np
import redis
from dotenv import load_dotenv
from datetime import datetime

# 載入 .env 檔案中的環境變數
load_dotenv()

# --- 設定常數 ---
BINANCE_API_URL = "https://api.binance.com/api/v3"
PIONEX_WEBHOOK_URL = "https://www.pionex.com/signal/api/v1/signal_listener/trading_view"
SYMBOL = "BNBUSDT"
INTERVAL = "2h"
INITIAL_CAPITAL = 100  # USDT
ORDER_QTY_PERCENT = 100 # 百分之百倉位
MACD_FAST_LEN = 12
MACD_SLOW_LEN = 26
MACD_SIGNAL_LEN = 9

# 從環境變數中獲取敏感資訊
PIONEX_TOKEN = os.getenv("PIONEX_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

# --- NN 權重與閾值 ---
# 將 Pine Script 中的輸入值設定為固定參數
NN_WEIGHTS = {
    "w1": [1, 0, -1, 0, 0, 1, 0, -1],
    "b1": [0, 0, 0, 0],
    "w2": [1, -1, -1, 1],
    "b2": 0.0,
    "threshold": 0.0,
}

# --- Redis 連線 ---
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print("成功連接到 Redis。")
except Exception as e:
    print(f"無法連接到 Redis: {e}")
    r = None

# --- Telegram 通知功能 ---
def send_telegram_message(message):
    """傳送訊息到 Telegram。"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Token 或 Chat ID 未設定，無法發送通知。")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"Pionex Bot 錯誤通知:\n{message}"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"發送 Telegram 訊息失敗: {e}")

# --- 技術指標計算函數 ---
def calculate_ema(data, length):
    """計算 EMA 指數移動平均。"""
    ema_list = [sum(data[:length]) / length]
    for price in data[length:]:
        ema_list.append(price * (2 / (length + 1)) + ema_list[-1] * (1 - 2 / (length + 1)))
    return ema_list[-1]

def calculate_macd(prices):
    """計算 MACD 線、訊號線、delta 和 delta_prev。"""
    if len(prices) < MACD_SLOW_LEN:
        return None, None, None, None

    fast_ema = calculate_ema(prices[-MACD_FAST_LEN:], MACD_FAST_LEN)
    slow_ema = calculate_ema(prices[-MACD_SLOW_LEN:], MACD_SLOW_LEN)
    
    macd_line = fast_ema - slow_ema

    # 從 Redis 獲取過去的 MACD_LINE 序列來計算 Signal Line
    macd_history_str = r.lrange("macd_history", 0, MACD_SIGNAL_LEN - 1) if r else []
    macd_history = [float(val) for val in macd_history_str]
    
    if len(macd_history) < MACD_SIGNAL_LEN:
        # 如果歷史數據不足，暫時無法計算 Signal Line
        return None, None, None, None

    # 將當前 MACD_LINE 加入歷史序列
    macd_history.insert(0, macd_line)
    
    signal_line = calculate_ema(macd_history, MACD_SIGNAL_LEN)
    
    delta = macd_line - signal_line
    delta_prev = float(r.get("delta_prev")) if r and r.exists("delta_prev") else delta
    
    # 更新 Redis 狀態
    if r:
        r.lpush("macd_history", macd_line)
        r.ltrim("macd_history", 0, MACD_SIGNAL_LEN)
        r.set("delta_prev", delta)

    return macd_line, signal_line, delta, delta_prev

# --- NN 核心邏輯 ---
def relu(x):
    """ReLU 激活函數。"""
    return max(0, x)

def nn_forward(delta, delta_prev, weights):
    """神經網路前向傳播。"""
    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]

    h0 = relu(w1[0] * delta + w1[1] * delta_prev + b1[0])
    h1 = relu(w1[2] * delta + w1[3] * delta_prev + b1[1])
    h2 = relu(w1[4] * delta + w1[5] * delta_prev + b1[2])
    h3 = relu(w1[6] * delta + w1[7] * delta_prev + b1[3])

    out = (h0 * w2[0] + h1 * w2[1] + h2 * w2[2] + h3 * w2[3] + b2)
    return out

# --- 交易訊號處理 ---
def get_current_position():
    """從 Redis 獲取當前持倉狀態。"""
    if r:
        return r.get("current_position") or "none"
    return "none"

def set_current_position(position):
    """更新 Redis 中的持倉狀態。"""
    if r:
        r.set("current_position", position)

def send_pionex_signal(action, contracts, symbol, price):
    """發送 Webhook 訊號到 Pionex。"""
    payload = {
        "data": {
            "action": action,
            "contracts": str(contracts),
            "position_size": "100%",
        },
        "price": str(price),
        "signal_param": {},
        "signal_type": "223269d3-1852-4de6-a0e8-a1e3ca5d39b9",
        "symbol": symbol,
        "time": str(int(time.time() * 1000)),
    }
    
    headers = {"Content-Type": "application/json"}
    webhook_url = f"{PIONEX_WEBHOOK_URL}?token={PIONEX_TOKEN}"
    
    try:
        response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功發送 {action} 訊號。")
    except requests.exceptions.RequestException as e:
        error_message = f"發送 Pionex 訊號失敗: {e}"
        print(error_message)
        send_telegram_message(error_message)

def initialize_data():
    """使用 K 線數據初始化價格序列和 Redis 狀態。"""
    print("正在獲取 K 線歷史數據以進行初始化...")
    try:
        # 獲取足夠的 K 線數據來初始化 MACD
        needed_klines = MACD_SLOW_LEN + MACD_SIGNAL_LEN + 10 # 額外多取一些
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": needed_klines}
        response = requests.get(f"{BINANCE_API_URL}/klines", params=params, timeout=10)
        response.raise_for_status()
        klines = response.json()
        
        close_prices = [float(kline[4]) for kline in klines]
        
        # 初始 MACD 計算，並填充 Redis
        macd_values = []
        for i in range(MACD_SLOW_LEN, len(close_prices)):
            # 這裡我們需要一個更準確的 EMA 計算
            fast_ema = np.mean(close_prices[i-MACD_FAST_LEN:i]) # 簡化為平均值
            slow_ema = np.mean(close_prices[i-MACD_SLOW_LEN:i]) # 簡化為平均值
            macd_values.append(fast_ema - slow_ema)

        if r:
            # 清空舊數據並寫入新數據
            r.delete("macd_history", "delta_prev", "current_position")
            r.rpush("macd_history", *macd_values[-MACD_SIGNAL_LEN:])
            print(f"已成功載入 {len(macd_values[-MACD_SIGNAL_LEN:])} 個 MACD 歷史值到 Redis。")

        return close_prices
        
    except Exception as e:
        error_message = f"初始化 K 線數據失敗: {e}"
        print(error_message)
        send_telegram_message(error_message)
        # 失敗後退出，因為沒有歷史數據將無法進行後續運算
        return None

def run_strategy():
    """主策略運行循環。"""
    prices = initialize_data()
    if prices is None:
        print("初始化失敗，機器人退出。")
        return

    while True:
        try:
            # 1. 獲取最新價格 (Tick)
            ticker_data = requests.get(f"{BINANCE_API_URL}/ticker/price?symbol={SYMBOL}", timeout=5).json()
            current_price = float(ticker_data["price"])

            # 2. 更新價格序列
            prices.append(current_price)
            prices.pop(0)

            # 3. 執行 MACD 與 NN 計算
            macd_line, signal_line, delta, delta_prev = calculate_macd(prices)
            
            if delta is None:
                print(f"資料不足，等待下一筆資料... (目前 MACD 歷史: {r.llen('macd_history')} 點)")
                time.sleep(1)
                continue

            nn_output = nn_forward(delta, delta_prev, NN_WEIGHTS)
            nn_signal = 0
            if nn_output > NN_WEIGHTS["threshold"]:
                nn_signal = 1
            elif nn_output < -NN_WEIGHTS["threshold"]:
                nn_signal = -1

            # 4. 判斷交易訊號並發送
            current_position = get_current_position()
            contracts_count = INITIAL_CAPITAL / current_price * (ORDER_QTY_PERCENT / 100)

            if nn_signal == 1 and current_position != "long":
                if current_position == "short":
                    send_pionex_signal("buy", contracts_count, SYMBOL, current_price)
                send_pionex_signal("buy", contracts_count, SYMBOL, current_price)
                set_current_position("long")
                print(f"--> {SYMBOL} 多頭訊號，當前價格: {current_price}")
            
            elif nn_signal == -1 and current_position != "short":
                if current_position == "long":
                    send_pionex_signal("sell", contracts_count, SYMBOL, current_price)
                send_pionex_signal("sell", contracts_count, SYMBOL, current_price)
                set_current_position("short")
                print(f"--> {SYMBOL} 空頭訊號，當前價格: {current_price}")

            time.sleep(1) # 每秒檢查一次

        except Exception as e:
            error_message = f"主循環發生錯誤: {e}"
            print(error_message)
            send_telegram_message(error_message)
            time.sleep(5)

if __name__ == "__main__":
    print("BNB MACD NN 交易機器人啟動...")
    run_strategy()

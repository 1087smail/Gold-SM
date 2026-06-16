import pandas as pd
import numpy as np
import requests
import time
import feedparser
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
import traceback
from binance.client import Client  # المكتبة الرسمية لبينانس

# ==========================================
# 🔑 إعدادات الحساب
# ==========================================
TELEGRAM_TOKEN = "8905827151:AAGX5wkJNqTR6BsakxgtJqooHnKcTYIvGcM"
TELEGRAM_CHAT_ID = "7502861613"

# ==========================================
# 📰 1. دالة جلب الأخبار والسياسات وتحليلها
# ==========================================
def get_news_sentiment():
    try:
        rss_url = "https://finance.yahoo.com/rss/headline?s=GC=F"
        feed = feedparser.parse(rss_url)
        sentiments = []
        titles_analyzed = []
        
        if feed.entries:
            for entry in feed.entries[:5]:
                title = entry.title
                analysis = TextBlob(title)
                sentiments.append(analysis.sentiment.polarity)
                titles_analyzed.append(title)
                
        if sentiments:
            return np.mean(sentiments), titles_analyzed
        return 0, ["No news headlines available currently"]
    except Exception as e:
        print(f"Warning: News analysis failed: {e}")
        return 0, ["Failed to fetch news"]

# ==========================================
# 🧠 2. دالة الذكاء الاصطناعي باستخدام بيانات Binance
# ==========================================
def train_and_predict_binance_ai():
    try:
        # الاتصال ببينانس بدون الحاجة لحساب أو مفاتيح سرية (للقراءة العامة فقط)
        client = Client()
        
        # جلب شموع التداول لعملة الذهب PAXG مقابل الدولار الرقمي USDT لآخر 30 يوم (كل شمعة ساعة)
        # هذا يعطينا 720 نقطة بيانات ممتازة جداً للتدريب اللحظي
        klines = client.get_historical_klines("PAXGUSDT", Client.KLINE_INTERVAL_1HOUR, "30 days ago UTC")
        
        if not klines:
            print("Error: Could not fetch data from Binance")
            return None, None, 0, []
            
        # تحويل البيانات إلى جدول منظم
        df = pd.DataFrame(klines, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'AssetVolume', 'Trades', 'BuyBase', 'BuyQuote', 'Ignore'])
        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        
        # هندسة الميزات (توقع شمعة الساعة القادمة بناءً على السابقة)
        df['Price_Lag'] = df['Close'].shift(1)
        df['Volume_Lag'] = df['Volume'].shift(1)
        df['Sentiment_Lag'] = np.random.uniform(-0.1, 0.2, len(df))
        df = df.dropna()
        
        X = df[['Price_Lag', 'Volume_Lag', 'Sentiment_Lag']]
        y = df['Close']
        
        # تدريب النموذج
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # تحليل الأخبار الحالية
        current_sentiment, latest_titles = get_news_sentiment()
        
        latest_price = df['Close'].iloc[-1]
        latest_volume = df['Volume'].iloc[-1]
        
        current_features = np.array([[latest_price, latest_volume, current_sentiment]])
        predicted_price = model.predict(current_features)[0]
        
        return round(latest_price, 2), round(predicted_price, 2), current_sentiment, latest_titles
        
    except Exception as e:
        print("Fatal error in Binance AI function:")
        traceback.print_exc()
        return None, None, 0, []

# ==========================================
# 💬 3. دالة إرسال الرسائل لتليجرام
# ==========================================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Message sent to Telegram successfully!")
        else:
            print(f"Telegram rejected the message. Code: {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to Telegram: {e}")

# ==========================================
# 🔄 4. المحرك الرئيسي
# ==========================================
def start_binance_bot():
    print("Binance Gold AI Bot has started...")
    last_sent_price = None
    
    while True:
        current_price, predicted_price, sentiment, titles = train_and_predict_binance_ai()
        
        if current_price and predicted_price:
            # منع التكرار إذا كان السعر ثابتاً تماماً على بينانس
            if last_sent_price is not None and current_price == last_sent_price:
                print(f"Binance Price is stable at ${current_price}. Skipping telegram message.")
            else:
                if sentiment > 0.05: news_status = "Positive and Supporting Upward Trend"
                elif sentiment < -0.05: news_status = "Negative and Driving Price Down"
                else: news_status = "Neutral and Calm"
                
                change_percent = ((predicted_price - current_price) / current_price) * 100
                direction = f"Expected Rise ({change_percent:.2f}%)" if predicted_price > current_price else f"Expected Drop ({abs(change_percent):.2f}%)"
                
                message = (
                    f"🎯 *تقرير ذهب Binance الذكي المحدث*\n\n"
                    f"💰 *سعر PAXG/USDT الحالي:* `${current_price}`\n"
                    f"🔮 *توقع الـ AI للساعة القادمة:* `${predicted_price}`\n"
                    f"📊 *الاتجاه المتوقع:* {direction}\n"
                    f"📰 *حالة السياسات والأخبار:* {news_status}\n\n"
                    f"📌 *أبرز العناوين السياسية والاقتصادية:*\n"
                )
                for i, title in enumerate(titles[:3], 1):
                    message += f"{i}. _{title}_\n"
                
                send_telegram_message(message)
                last_sent_price = current_price
        else:
            print("Warning: Skipping this cycle due to data fetching failure.")
            
        print("Waiting for the next hourly check...")
        time.sleep(3600)

if __name__ == "__main__":
    start_binance_bot()
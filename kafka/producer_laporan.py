import time
import json
import feedparser
import hashlib
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8') if k else None
)

TOPIC = 'smartflood-laporan'
RSS_URL = "https://rss.tempo.co/tag/cuaca"
sent_articles = set() # Menghindari duplikat

def fetch_rss():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Mengecek RSS Feed Tempo Cuaca...")
    try:
        feed = feedparser.parse(RSS_URL)
        new_count = 0
        
        for entry in feed.entries[:5]: # Ambil 5 berita terbaru
            url_hash = hashlib.md5(entry.link.encode()).hexdigest()
            
            if url_hash not in sent_articles:
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "judul": entry.title,
                    "link": entry.link,
                    "sumber": "Tempo RSS"
                }
                
                producer.send(TOPIC, key=url_hash, value=payload)
                sent_articles.add(url_hash)
                new_count += 1
                print(f"✅ Berita baru terkirim: {entry.title}")
                
        producer.flush()
        if new_count == 0:
            print("Tidak ada berita baru.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Mulai menjalankan Producer Laporan RSS...")
    while True:
        fetch_rss()
        # Sleep 5 menit untuk RSS
        time.sleep(300)

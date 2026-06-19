import time
import json
import requests
from kafka import KafkaProducer
from datetime import datetime

# Inisialisasi Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8') if k else None
)

TOPIC = 'smartflood-cuaca'
# Koordinat pusat Surabaya
API_URL = "https://api.open-meteo.com/v1/forecast?latitude=-7.2504&longitude=112.7688&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=Asia%2FJakarta"

def fetch_weather():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Mengambil data cuaca Open-Meteo...")
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            
            # Format payload untuk dikirim ke Kafka
            payload = {
                "timestamp": current['time'],
                "suhu_c": current['temperature_2m'],
                "kelembaban_pct": current['relative_humidity_2m'],
                "curah_hujan_mm": current['precipitation'],
                "kecepatan_angin_ms": current['wind_speed_10m'] / 3.6, # Convert km/h to m/s
                "lokasi": "Surabaya"
            }
            
            # Kirim ke Kafka
            producer.send(TOPIC, key="SBY", value=payload)
            producer.flush()
            print(f"✅ Data terkirim ke Kafka: {payload}")
        else:
            print(f"❌ Gagal mengambil API: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Mulai menjalankan Producer Cuaca...")
    while True:
        fetch_weather()
        # Sleep 60 detik sebelum request lagi
        time.sleep(60)

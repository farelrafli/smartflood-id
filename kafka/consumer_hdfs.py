import json
import os
import subprocess
from datetime import datetime
from kafka import KafkaConsumer

# Koneksi ke Kafka (Peringatan value_deserializer bisa diabaikan)
consumer = KafkaConsumer(
    'smartflood-cuaca', 'smartflood-laporan',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='hdfs-writer-group',
    auto_offset_reset='latest'
)

print("Consumer aktif. Menunggu data dari Kafka...")

for message in consumer:
    topic = message.topic
    data = message.value
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. Simpan salinan terbaru ke Dashboard Lokal (untuk Real-time UI)
    local_path = f"dashboard/data/live_{topic.split('-')[1]}.json"
    with open(local_path, "w") as f:
        json.dump(data, f)
        
    # 2. Simpan ke HDFS menggunakan subprocess + docker exec
    hdfs_folder = f"/data/smartflood/{topic.split('-')[1]}/"
    hdfs_file = f"{hdfs_folder}{timestamp_str}.json"
    
    try:
        # Convert JSON ke string lalu ubah jadi bytes
        json_bytes = json.dumps(data).encode('utf-8')
        
        # Jalankan command: docker exec -i namenode hdfs dfs -put - /data/...
        # Tanda "-" berarti hdfs akan membaca file dari input yang kita tembakkan (json_bytes)
        cmd = ["docker", "exec", "-i", "namenode", "hdfs", "dfs", "-put", "-", hdfs_file]
        
        result = subprocess.run(cmd, input=json_bytes, capture_output=True)
        
        if result.returncode == 0:
            print(f"✅ [{topic}] Event tersimpan ke HDFS: {hdfs_file}")
        else:
            print(f"❌ Gagal HDFS: {result.stderr.decode('utf-8')}")
            
    except Exception as e:
        print(f"❌ Error saat eksekusi Docker: {e}")

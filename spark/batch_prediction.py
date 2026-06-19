"""
SmartFlood ID — Apache Spark Batch Analytics
Arsitektur Data Lakehouse (Medallion Pattern):
  Bronze → Silver → Gold

Bronze : Raw JSON dari HDFS (output Kafka consumer)
Silver : Cleaned + feature-engineered Parquet (Spark SQL)
Gold   : Prediksi final Parquet + JSON untuk Streamlit dashboard
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

# ==============================================================
# 1. INISIALISASI SPARK SESSION
# ==============================================================
spark = SparkSession.builder \
    .appName("SmartFlood_Lakehouse_Analytics") \
    .master("local[*]") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ SparkSession berhasil dibuat — SmartFlood Lakehouse v2")

# ==============================================================
# 2. LOAD MODEL RF (dari file lokal, disebar ke Spark jika perlu)
# ==============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'smartflood_rf_model.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '..', 'smartflood_encoder.pkl')

rf_model = None
le_encoder = None
try:
    rf_model = joblib.load(MODEL_PATH)
    le_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model RF + LabelEncoder berhasil dimuat.")
except Exception as e:
    print(f"⚠️ Model tidak ditemukan, fallback ke formula heuristik: {e}")

# ==============================================================
# 3. BRONZE LAYER — BACA RAW JSON DARI HDFS
# ==============================================================
print("\n📥 [BRONZE] Membaca raw data dari HDFS...")

try:
    df_cuaca_raw = spark.read.json("hdfs://localhost:8020/data/smartflood/cuaca/*.json")
    df_cuaca_raw.createOrReplaceTempView("bronze_cuaca")
    cuaca_count = df_cuaca_raw.count()
    print(f"✅ Bronze cuaca: {cuaca_count} records dari HDFS")
    
    latest_weather = spark.sql("""
        SELECT curah_hujan_mm, kelembaban_pct, suhu_c, kecepatan_angin_ms, timestamp
        FROM bronze_cuaca
        ORDER BY timestamp DESC
        LIMIT 1
    """).collect()[0]
    
    rr    = float(latest_weather['curah_hujan_mm'] or 0)
    rh    = float(latest_weather['kelembaban_pct'] or 80)
    tavg  = float(latest_weather['suhu_c'] or 28.0)
    wind  = float(latest_weather['kecepatan_angin_ms'] or 3.0)
    latest_time = str(latest_weather['timestamp'] or datetime.now().isoformat())
    data_source = "HDFS Bronze Layer"

except Exception as e:
    print(f"⚠️ HDFS belum tersedia atau kosong ({e}). Fallback ke live_cuaca.json lokal.")
    local_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'data', 'live_cuaca.json')
    if os.path.exists(local_path):
        with open(local_path) as f:
            live = json.load(f)
        rr    = float(live.get('curah_hujan_mm', 15.0))
        rh    = float(live.get('kelembaban_pct', 80))
        tavg  = float(live.get('suhu_c', 28.0))
        wind  = float(live.get('kecepatan_angin_ms', 3.0))
        latest_time = live.get('timestamp', datetime.now().isoformat())
    else:
        rr, rh, tavg, wind, latest_time = 15.0, 80, 28.0, 3.0, datetime.now().isoformat()
    data_source = "Local fallback (live_cuaca.json)"

print(f"   Curah hujan referensi: {rr} mm | Suhu: {tavg} °C | Angin: {wind:.2f} m/s")

# ==============================================================
# 4. SILVER LAYER — FEATURE ENGINEERING VIA SPARK SQL
# ==============================================================
print("\n⚙️  [SILVER] Feature engineering dengan Spark SQL...")

HIDROLOGI = [
    {'kecamatan':'Benowo',     'elevasi_m':3.2,  'kapasitas_drainase_m3s':15, 'luas_km2':24.8, 'penduduk':72541,  'koef_limpasan':0.65, 'jarak_pantai_km':12.5, 'frekuensi_historis':8},
    {'kecamatan':'Pakal',      'elevasi_m':4.1,  'kapasitas_drainase_m3s':18, 'luas_km2':22.5, 'penduduk':63218,  'koef_limpasan':0.60, 'jarak_pantai_km':15.2, 'frekuensi_historis':6},
    {'kecamatan':'Tandes',     'elevasi_m':5.0,  'kapasitas_drainase_m3s':22, 'luas_km2':9.7,  'penduduk':108432, 'koef_limpasan':0.72, 'jarak_pantai_km':8.3,  'frekuensi_historis':10},
    {'kecamatan':'Lakarsantri','elevasi_m':8.5,  'kapasitas_drainase_m3s':35, 'luas_km2':18.4, 'penduduk':57819,  'koef_limpasan':0.45, 'jarak_pantai_km':20.1, 'frekuensi_historis':3},
    {'kecamatan':'Wonokromo',  'elevasi_m':6.2,  'kapasitas_drainase_m3s':28, 'luas_km2':8.5,  'penduduk':143265, 'koef_limpasan':0.80, 'jarak_pantai_km':9.8,  'frekuensi_historis':12},
    {'kecamatan':'Rungkut',    'elevasi_m':9.8,  'kapasitas_drainase_m3s':40, 'luas_km2':21.1, 'penduduk':95847,  'koef_limpasan':0.70, 'jarak_pantai_km':18.5, 'frekuensi_historis':4},
    {'kecamatan':'Sukolilo',   'elevasi_m':7.3,  'kapasitas_drainase_m3s':32, 'luas_km2':23.5, 'penduduk':87631,  'koef_limpasan':0.65, 'jarak_pantai_km':14.2, 'frekuensi_historis':5},
    {'kecamatan':'Kenjeran',   'elevasi_m':2.8,  'kapasitas_drainase_m3s':12, 'luas_km2':7.6,  'penduduk':75234,  'koef_limpasan':0.75, 'jarak_pantai_km':1.2,  'frekuensi_historis':15},
    {'kecamatan':'Bulak',      'elevasi_m':2.5,  'kapasitas_drainase_m3s':10, 'luas_km2':6.8,  'penduduk':42187,  'koef_limpasan':0.78, 'jarak_pantai_km':0.8,  'frekuensi_historis':18},
    {'kecamatan':'Semampir',   'elevasi_m':3.5,  'kapasitas_drainase_m3s':14, 'luas_km2':8.9,  'penduduk':132541, 'koef_limpasan':0.82, 'jarak_pantai_km':2.1,  'frekuensi_historis':14},
    {'kecamatan':'Bubutan',    'elevasi_m':4.2,  'kapasitas_drainase_m3s':20, 'luas_km2':4.8,  'penduduk':98765,  'koef_limpasan':0.85, 'jarak_pantai_km':6.5,  'frekuensi_historis':9},
    {'kecamatan':'Simokerto',  'elevasi_m':4.8,  'kapasitas_drainase_m3s':18, 'luas_km2':5.3,  'penduduk':112340, 'koef_limpasan':0.83, 'jarak_pantai_km':4.3,  'frekuensi_historis':11},
    {'kecamatan':'Sawahan',    'elevasi_m':5.1,  'kapasitas_drainase_m3s':24, 'luas_km2':9.1,  'penduduk':187654, 'koef_limpasan':0.78, 'jarak_pantai_km':7.8,  'frekuensi_historis':10},
    {'kecamatan':'Genteng',    'elevasi_m':6.0,  'kapasitas_drainase_m3s':26, 'luas_km2':4.3,  'penduduk':76543,  'koef_limpasan':0.80, 'jarak_pantai_km':8.9,  'frekuensi_historis':7},
    {'kecamatan':'Gubeng',     'elevasi_m':5.5,  'kapasitas_drainase_m3s':30, 'luas_km2':7.8,  'penduduk':145231, 'koef_limpasan':0.75, 'jarak_pantai_km':10.2, 'frekuensi_historis':8},
]

bulan = datetime.now().month
musim = 1 if bulan in [11,12,1,2,3,4] else 0
ss = 4.0 if rr > 10 else 8.0
durasi = max(0, round((8-ss)*min(rr/20,1)))
rr_3d = rr * 1.5 + 20
pasang_base = 0.8 + 0.4*np.sin(2*np.pi*bulan/12)

FEATURES = ['curah_hujan_mm','curah_hujan_3hari','kelembaban_pct','suhu_c',
            'kecepatan_angin','durasi_hujan_jam','bulan','musim_hujan',
            'elevasi_m','koef_limpasan','frekuensi_historis','kecamatan_enc',
            'tma_sungai_m','pasang_surut_m','excess_drainase','hujan_ekstrem']

results = []
for kec in HIDROLOGI:
    debit_lr = kec['koef_limpasan'] * (rr/3600) * kec['luas_km2'] * 1000
    excess   = max(0, debit_lr - kec['kapasitas_drainase_m3s'])
    pasang   = max(0, pasang_base - kec['jarak_pantai_km'] * 0.05)
    
    tinggi_gen = np.clip((excess/kec['kapasitas_drainase_m3s'])*0.5 + pasang*0.3 + (5-kec['elevasi_m'])*0.1, 0.05, 3).round(2)
    luas_td    = np.clip(kec['luas_km2']*(tinggi_gen/2)*0.6, 0.1, kec['luas_km2']).round(2)
    ptd        = int((kec['penduduk']/kec['luas_km2']) * luas_td)
    lead       = max(1, round(12 - (rr/20) - (5-kec['elevasi_m'])*0.5))
    
    # Gunakan RF model jika tersedia, fallback ke heuristik
    if rf_model is not None and le_encoder is not None:
        try:
            tma_sungai = float(np.clip(rr/25*(1+(5-kec['elevasi_m'])/5), 0.2, 5).round(2))
            feat_dict = {
                'curah_hujan_mm': rr, 'curah_hujan_3hari': rr_3d, 'kelembaban_pct': rh,
                'suhu_c': tavg, 'kecepatan_angin': wind, 'durasi_hujan_jam': durasi,
                'bulan': bulan, 'musim_hujan': musim, 'elevasi_m': kec['elevasi_m'],
                'koef_limpasan': kec['koef_limpasan'], 'frekuensi_historis': kec['frekuensi_historis'],
                'kecamatan_enc': le_encoder.transform([kec['kecamatan']])[0],
                'tma_sungai_m': tma_sungai, 'pasang_surut_m': round(pasang, 3),
                'excess_drainase': round(excess, 3), 'hujan_ekstrem': 1 if rr > 50 else 0
            }
            df_feat = pd.DataFrame([feat_dict])[FEATURES]
            prob = float(rf_model.predict_proba(df_feat)[0][1] * 100)
            method = "RF Model"
        except:
            # fallback heuristik
            indeks = rr*0.25 + (rr_3d/3)*0.20 + (10-kec['elevasi_m'])*0.15 + excess*0.15 + pasang*0.10
            prob = min(100.0, (indeks / 5.0) * 100)
            method = "Heuristic fallback"
    else:
        indeks = rr*0.25 + (rr_3d/3)*0.20 + (10-kec['elevasi_m'])*0.15 + excess*0.15 + pasang*0.10
        prob = min(100.0, (indeks / 5.0) * 100)
        method = "Heuristic (no model)"
    
    status = '🔴 KRITIS' if prob > 70 else '🟡 WASPADA' if prob > 40 else '🟢 AMAN'
    
    results.append({
        'Kecamatan': kec['kecamatan'],
        'Status': status,
        'Probabilitas': round(prob, 1),
        'Tinggi_Genangan_m': float(tinggi_gen),
        'Luas_Terdampak_km2': float(luas_td),
        'Penduduk_Terdampak': ptd,
        'Lead_Time_jam': lead,
        'Method': method
    })

df_results = pd.DataFrame(results).sort_values('Probabilitas', ascending=False)
print(f"✅ Silver layer: {len(df_results)} kecamatan diproses via {results[0]['Method']}")

# ==============================================================
# 5. GOLD LAYER — SIMPAN KE HDFS (PARQUET) & LOKAL (JSON)
# ==============================================================
print("\n💾 [GOLD] Menyimpan hasil ke Gold Layer...")

# Simpan ke HDFS sebagai Parquet (Gold layer) — DILAKUKAN SEBELUM nulis JSON lokal
# supaya hasil partisi nyata bisa disertakan sebagai bukti, bukan klaim kosong.
gold_hdfs_path = "hdfs://localhost:8020/data/smartflood/hasil/prediksi_latest"
partition_proof = {"written": False, "path": gold_hdfs_path, "partitions": [], "error": None}
try:
    df_spark_gold = spark.createDataFrame(df_results.drop(columns=['Method']))
    df_spark_gold = df_spark_gold.withColumn("tanggal_proses", F.current_date())
    df_spark_gold.write.mode("overwrite").partitionBy("tanggal_proses").parquet(gold_hdfs_path)
    print(f"✅ Gold layer HDFS: {gold_hdfs_path} (Parquet, partisi tanggal)")

    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path_obj = spark._jvm.org.apache.hadoop.fs.Path(gold_hdfs_path)
    statuses = fs.listStatus(path_obj)
    partition_dirs = [s.getPath().getName() for s in statuses if s.isDirectory()]
    partition_proof["written"] = True
    partition_proof["partitions"] = partition_dirs
    print(f"   Partisi terbukti ada: {partition_dirs}")
except Exception as e:
    print(f"⚠️ Gagal menyimpan ke HDFS Gold layer: {e}")
    partition_proof["error"] = str(e)

output_payload = {
    "timestamp_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_source": data_source,
    "cuaca_referensi": {
        "waktu": latest_time,
        "curah_hujan": float(rr),
        "suhu": float(tavg),
        "angin": float(wind)
    },
    "prediksi_kecamatan": df_results.to_dict(orient='records'),
    "hdfs_partition_proof": partition_proof
}

# Simpan lokal (dikonsumsi Streamlit dashboard)
os.makedirs("dashboard/data", exist_ok=True)
local_out = "dashboard/data/spark_results.json"
with open(local_out, "w") as f:
    json.dump(output_payload, f, indent=4)
print(f"✅ Gold layer lokal: {local_out}")

# ==============================================================
# 6. SPARK SQL SUMMARY (untuk demo)
# ==============================================================
df_spark_gold_local = spark.createDataFrame(df_results)
df_spark_gold_local.createOrReplaceTempView("gold_prediksi")

print("\n📊 [SPARK SQL] Summary Analytics:")
spark.sql("""
    SELECT Status,
           COUNT(*) as jumlah_kecamatan,
           ROUND(AVG(Probabilitas), 2) as avg_probabilitas,
           SUM(Penduduk_Terdampak) as total_terdampak
    FROM gold_prediksi
    GROUP BY Status
    ORDER BY avg_probabilitas DESC
""").show()

print("\n🏆 Top 5 Kecamatan Risiko Tertinggi (Spark SQL):")
spark.sql("""
    SELECT Kecamatan, Status, Probabilitas, Tinggi_Genangan_m, Penduduk_Terdampak, Lead_Time_jam
    FROM gold_prediksi
    ORDER BY Probabilitas DESC
    LIMIT 5
""").show()

spark.stop()
print("\n✅ SmartFlood Spark Lakehouse selesai.")

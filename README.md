# 🌊 SmartFlood ID — Sistem Prediksi & Peringatan Dini Banjir Berbasis Big Data
**Kota Surabaya | Gemastik 2026 — Smart City Track**

---

## 🎯 Identifikasi Masalah (5V Big Data)
Surabaya menghadapi ancaman banjir tahunan yang berdampak pada ratusan ribu warga. Sistem peringatan dini eksisting (BPBD, PetaBencana) bersifat **reaktif** — hanya merespons banjir yang sudah terjadi, tanpa prediksi berbasis data untuk tindakan preventif.

| Dimensi | Detail |
|---------|--------|
| **Volume** | 527 hari data BMKG Stasiun Juanda (Nov 2024 – Mei 2026), 15 kecamatan, ribuan rekaman fitur |
| **Velocity** | Ingestion real-time via Apache Kafka (polling Open-Meteo tiap 60 detik, RSS tiap 5 menit) |
| **Variety** | Data cuaca API (Open-Meteo), laporan warga (Tempo RSS), data hidrologi statis, ground truth BPBD |
| **Veracity** | 5-Fold Time-Series CV untuk validasi, threshold F1.5 optimal, transparansi skor (AUC 0.75, jujur) |
| **Value** | Prediksi probabilitas banjir per kecamatan + estimasi tinggi genangan, luas terdampak, jiwa terdampak, lead time peringatan dini |

---

## 🏗️ Arsitektur Big Data (4-Layer)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMARTFLOOD ID — BIG DATA PIPELINE                │
└─────────────────────────────────────────────────────────────────────┘

  LAYER 1: INGESTION
  ┌─────────────────┐    ┌─────────────────┐
  │  Open-Meteo API │    │  Tempo RSS Feed │
  │  (cuaca live)   │    │  (laporan berita)│
  └────────┬────────┘    └────────┬────────┘
           │                     │
           ▼                     ▼
  ┌─────────────────────────────────────────┐
  │        Apache Kafka (2 Topics)          │
  │  📤 smartflood-cuaca                   │
  │  📤 smartflood-laporan                 │
  └──────────────────┬──────────────────────┘
                     │
  LAYER 2: STORAGE   │
                     ▼
  ┌─────────────────────────────────────────┐
  │          HDFS (Namenode Docker)         │
  │  🥉 Bronze: /data/smartflood/cuaca/    │
  │  🥉 Bronze: /data/smartflood/laporan/  │
  └──────────────────┬──────────────────────┘
                     │
  LAYER 3: PROCESSING│
                     ▼
  ┌─────────────────────────────────────────┐
  │       Apache Spark (batch_prediction)   │
  │  🥈 Silver: Spark SQL cleansing         │
  │            + feature engineering        │
  │  🥇 Gold: ML prediction (RF model)     │
  │            → Parquet (partisi tanggal)  │
  └──────────────────┬──────────────────────┘
                     │
  LAYER 4: SERVING   │
                     ▼
  ┌─────────────────────────────────────────┐
  │     Streamlit Dashboard (app.py)        │
  │  📍 Analisis Prediksi per Kecamatan    │
  │  🗺️ Peta Risiko Interaktif (Folium)    │
  │  🔬 Clustering K-Means (Zona Risiko)   │
  │  ⚡ Spark Batch Analytics              │
  │  📡 Event Kafka Monitoring             │
  │  📊 Evaluasi Model                     │
  └─────────────────────────────────────────┘
```

---

## ✨ Fitur Utama

### 📍 Prediksi Real-Time (RF + Kafka)
- Probabilitas banjir per 15 kecamatan Surabaya
- Model Random Forest, 5-Fold Time-Series CV, AUC 0.75
- Estimasi tinggi genangan (m), luas terdampak (km²), % wilayah
- Estimasi penduduk terdampak (jiwa)
- Lead time peringatan dini (jam)
- **Flood Risk Score** komposit (probabilitas + genangan + populasi + urgensi)

### 🗺️ Peta Risiko Interaktif (Folium + OpenStreetMap)
- Visualisasi geospasial risiko per kecamatan
- Marker proporsional dengan Risk Score
- Popup interaktif dengan detail lengkap tiap kecamatan
- Color-coded: Merah/Kuning/Hijau sesuai status

### 🔬 K-Means Clustering Zona Risiko (Teknik Analisis #2)
- Clustering 15 kecamatan ke 3 zona: Kritis / Sedang / Aman
- Fitur multidimensi: probabilitas, genangan, kepadatan, drainase, elevasi
- Elbow method untuk pemilihan K optimal
- Scatter plot probabilitas vs Risk Score dengan label cluster
- Statistik agregat per zona

### ⚡ Spark Batch Analytics (Lakehouse Integration)
- **Bronze → Silver → Gold** (Medallion Architecture)
- RF model terintegrasi penuh ke Spark (bukan heuristik)
- Output Parquet terpartisi per tanggal di HDFS Gold layer
- Perbandingan prediksi Spark batch vs live RF di dashboard

---

## 🚀 Cara Menjalankan

### Prerequisites
```bash
# 1. Docker (untuk HDFS + Kafka)
docker-compose up -d

# 2. Install dependencies Python
pip install -r requirements.txt
```

### Jalankan Pipeline

```bash
# Terminal 1 — Kafka Producer Cuaca
python kafka/producer_cuaca.py

# Terminal 2 — Kafka Producer Laporan RSS
python kafka/producer_laporan.py

# Terminal 3 — Kafka Consumer → HDFS
python kafka/consumer_hdfs.py

# Terminal 4 — Spark Batch (periodik, mis. tiap 1 jam)
python spark/batch_prediction.py

# Terminal 5 — Streamlit Dashboard
streamlit run app.py
```

---

## 📊 Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| AUC (Time-Series CV) | 0.7511 |
| Akurasi | 93.00% |
| F1-Score (Banjir) | 0.31 |
| Fold terbaik | Fold 4 (AUC 0.9029) |
| Threshold optimal | 0.465 (F1.5-Score) |

**Catatan transparansi:** F1 kelas "Banjir" rendah (0.31) disebabkan ketidakseimbangan data ekstrem (71 kejadian banjir dari 1581 total). Oversampling SMOTE atau cost-sensitive learning dapat meningkatkan recall kelas minoritas. Ground truth 30 hari banjir bersumber dari BPBD Surabaya dikombinasikan dengan rule threshold curah hujan kumulatif 3 hari >80mm.

---

## 📁 Struktur Proyek

```
smartflood-surabaya-main/
├── app.py                         ← Dashboard Streamlit (4 layer terintegrasi)
├── requirements.txt
├── style.css
├── smartflood_rf_model.pkl        ← Model Random Forest terlatih
├── smartflood_encoder.pkl         ← LabelEncoder kecamatan
├── aug.png / importance.png / data.png ← Gambar evaluasi
├── kafka/
│   ├── producer_cuaca.py          ← Kafka Producer (Open-Meteo)
│   ├── producer_laporan.py        ← Kafka Producer (Tempo RSS)
│   └── consumer_hdfs.py           ← Kafka Consumer → HDFS Bronze
├── spark/
│   └── batch_prediction.py        ← Spark Lakehouse (Bronze→Silver→Gold)
└── dashboard/
    └── data/
        ├── live_cuaca.json        ← Output Kafka consumer (cuaca)
        ├── live_laporan.json      ← Output Kafka consumer (laporan RSS)
        └── spark_results.json     ← Output Spark Gold layer (lokal)
```

---

## 👥 Tim
**Kelompok SmartFlood ID** | Institut Teknologi Sepuluh Nopember (ITS)  
Program Studi Teknologi Informasi | Big Data — Kelas B

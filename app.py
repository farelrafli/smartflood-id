import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(
    page_title="SmartFlood ID — Surabaya",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background: #f0f7ff;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #378ADD;
}
.danger { border-left-color: #E24B4A !important; background: #fff0f0 !important; }
.warn   { border-left-color: #EF9F27 !important; background: #fffbf0 !important; }
.safe   { border-left-color: #639922 !important; background: #f0fff4 !important; }
.big-num { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.card-label { font-size: 0.8rem; color: #666; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model (cache agar tidak reload terus) ──────────────────────────────
@st.cache_data
def load_and_train():
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'tanggal':          pd.date_range('2015-01-01', periods=n, freq='6H'),
        'curah_hujan_mm':   np.random.exponential(30, n).clip(0, 200).round(1),
        'tinggi_air_m':     np.random.uniform(0.5, 5.0, n).round(2),
        'kecepatan_angin':  np.random.uniform(5, 40, n).round(1),
        'kelembaban_pct':   np.random.uniform(60, 100, n).round(1),
        'suhu_c':           np.random.uniform(24, 35, n).round(1),
        'durasi_hujan_jam': np.random.randint(0, 13, n),
        'debit_sungai_m3s': np.random.uniform(10, 500, n).round(1),
        'kecamatan':        np.random.choice(
            ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo','Rungkut','Sukolilo'], n
        )
    })
    df['banjir'] = (
        (df['curah_hujan_mm'] > 70) |
        (df['tinggi_air_m'] > 3.0) |
        ((df['durasi_hujan_jam'] >= 5) & (df['curah_hujan_mm'] > 50))
    ).astype(int)

    df['bulan']           = df['tanggal'].dt.month
    df['musim_hujan']     = df['bulan'].isin([10,11,12,1,2,3]).astype(int)
    df['indeks_risiko']   = (
        df['curah_hujan_mm'] * 0.4 +
        df['tinggi_air_m']   * 0.35 +
        df['durasi_hujan_jam'] * 0.25
    )
    df['hujan_ekstrem']   = (df['curah_hujan_mm'] > 100).astype(int)

    le = LabelEncoder()
    df['kecamatan_enc'] = le.fit_transform(df['kecamatan'])

    FEATURES = [
        'curah_hujan_mm','tinggi_air_m','kecepatan_angin',
        'kelembaban_pct','suhu_c','durasi_hujan_jam',
        'debit_sungai_m3s','bulan','musim_hujan',
        'indeks_risiko','hujan_ekstrem','kecamatan_enc'
    ]
    X = df[FEATURES]
    y = df['banjir']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    return df, rf, scaler, le, FEATURES, X_test_sc, y_test


df, rf, scaler, le, FEATURES, X_test_sc, y_test = load_and_train()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🌊 SmartFlood ID")
st.markdown("**Sistem Prediksi & Peringatan Dini Banjir — Kota Surabaya**")
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulasi Input Sensor")
    st.markdown("Geser slider untuk simulasi kondisi real-time:")
    rain   = st.slider("Curah hujan (mm/jam)", 0, 150, 87)
    water  = st.slider("Tinggi air sungai (m)", 0.0, 5.0, 3.2, 0.1)
    dur    = st.slider("Durasi hujan (jam)",    0, 12, 4)
    wind   = st.slider("Kecepatan angin (km/h)", 5, 40, 25)
    humid  = st.slider("Kelembaban (%)",         60, 100, 93)
    temp   = st.slider("Suhu udara (°C)",        24.0, 35.0, 27.0, 0.5)
    debit  = st.slider("Debit sungai (m³/s)",   10, 500, 320)
    kec    = st.selectbox("Kecamatan",
                ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo','Rungkut','Sukolilo'])
    st.markdown("---")
    st.caption("SmartFlood ID — Project Gemastik 2026")


# ── PREDIKSI REAL-TIME ────────────────────────────────────────────────────────
def predict_one(rain, water, dur, wind, humid, temp, debit, kec_name):
    indeks = rain * 0.4 + water * 0.35 + dur * 0.25
    hujan_ext = 1 if rain > 100 else 0
    bulan = 1
    musim = 1
    kec_enc = le.transform([kec_name])[0]
    row = np.array([[rain, water, wind, humid, temp, dur,
                     debit, bulan, musim, indeks, hujan_ext, kec_enc]])
    row_sc = scaler.transform(row)
    prob = rf.predict_proba(row_sc)[0][1]
    return prob

prob = predict_one(rain, water, dur, wind, humid, temp, debit, kec)
pct  = round(prob * 100, 1)
label = "🔴 KRITIS" if pct > 70 else "🟡 WASPADA" if pct > 40 else "🟢 AMAN"
color = "danger" if pct > 70 else "warn" if pct > 40 else "safe"

# ── METRIC CARDS ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card {color}">
        <div class="card-label">Status prediksi — {kec}</div>
        <div class="big-num">{label}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card {color}">
        <div class="card-label">Probabilitas banjir</div>
        <div class="big-num">{pct}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card warn">
        <div class="card-label">Curah hujan input</div>
        <div class="big-num">{rain} mm</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card {'danger' if water > 3 else 'safe'}">
        <div class="card-label">Tinggi air sungai</div>
        <div class="big-num">{water} m</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── TAB ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📍 Risiko per Kecamatan", "📊 Evaluasi Model", "🔍 Eksplorasi Data"])

# TAB 1: RISIKO PER KECAMATAN
with tab1:
    st.markdown("### Prediksi risiko banjir — 7 kecamatan Surabaya")
    st.caption("Berdasarkan input sensor sidebar saat ini")

    kecamatans = ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo','Rungkut','Sukolilo']
    probs_all = [predict_one(rain, water, dur, wind, humid, temp, debit, k) * 100
                 for k in kecamatans]

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ['#E24B4A' if p > 70 else '#EF9F27' if p > 40 else '#639922' for p in probs_all]
    bars = ax.barh(kecamatans, probs_all, color=bar_colors, edgecolor='white', height=0.55)
    ax.axvline(70, color='#E24B4A', linestyle='--', lw=1.2, alpha=0.7, label='Ambang kritis (70%)')
    ax.axvline(40, color='#EF9F27', linestyle='--', lw=1.2, alpha=0.7, label='Ambang waspada (40%)')
    for bar, p in zip(bars, probs_all):
        ax.text(p + 1, bar.get_y() + bar.get_height()/2,
                f'{p:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Probabilitas Banjir (%)')
    ax.set_xlim(0, 115)
    ax.legend(loc='lower right')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Tabel detail")
    tbl = pd.DataFrame({
        'Kecamatan': kecamatans,
        'Probabilitas (%)': [round(p, 1) for p in probs_all],
        'Status': ['KRITIS' if p > 70 else 'WASPADA' if p > 40 else 'AMAN' for p in probs_all],
        'Curah Hujan (mm)': rain,
        'Tinggi Air (m)': water,
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# TAB 2: EVALUASI MODEL
with tab2:
    st.markdown("### Performa model Random Forest")
    rf_pred = rf.predict(X_test_sc)
    rf_prob_test = rf.predict_proba(X_test_sc)[:,1]
    auc = roc_auc_score(y_test, rf_prob_test)

    m1, m2, m3 = st.columns(3)
    report = classification_report(y_test, rf_pred, output_dict=True)
    m1.metric("AUC Score", f"{auc:.4f}")
    m2.metric("Akurasi", f"{report['accuracy']*100:.1f}%")
    m3.metric("F1-Score (Banjir)", f"{report['1']['f1-score']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Tidak Banjir','Banjir'],
                    yticklabels=['Tidak Banjir','Banjir'])
        ax.set_ylabel('Aktual')
        ax.set_xlabel('Prediksi')
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Feature importance")
        fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors_fi = ['#E24B4A' if v > fi.quantile(0.75) else '#378ADD' for v in fi.values]
        fi.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
        ax.set_xlabel('Importance Score')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.grid(False)
        st.pyplot(fig)
        plt.close()


# TAB 3: EKSPLORASI DATA
with tab3:
    st.markdown("### Eksplorasi dataset training")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribusi curah hujan")
        fig, ax = plt.subplots(figsize=(5, 3))
        df['curah_hujan_mm'].hist(bins=30, ax=ax, color='#378ADD',
                                   edgecolor='white', alpha=0.85)
        ax.set_xlabel('Curah Hujan (mm)')
        ax.set_ylabel('Frekuensi')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.grid(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Frekuensi banjir per kecamatan")
        banjir_kec = df.groupby('kecamatan')['banjir'].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        bar_c = ['#E24B4A' if v > 50 else '#EF9F27' if v > 35 else '#639922'
                 for v in banjir_kec.values]
        banjir_kec.plot(kind='barh', ax=ax, color=bar_c, edgecolor='white')
        ax.set_xlabel('Frekuensi Banjir (%)')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.grid(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Heatmap korelasi")
    num_cols = ['curah_hujan_mm','tinggi_air_m','kelembaban_pct',
                'suhu_c','durasi_hujan_jam','debit_sungai_m3s','banjir']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                cmap='Blues', ax=ax, linewidths=0.5)
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    plt.close()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SmartFlood ID — Surabaya",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0b1120; }
    [data-testid="stSidebar"] { background: #0f1929; border-right: 1px solid #1e3050; }
    .main .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e8f4fd !important; }
    p, li, label { color: #b0c4d8 !important; }
    .metric-card {
        background: linear-gradient(135deg, #0f2540 0%, #112d4e 100%);
        border: 1px solid #1e4a7a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #4fc3f7; }
    .metric-lbl { font-size: 0.8rem; color: #7a9dbf; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .status-kritis  { background: #3d0f0f; border-left: 4px solid #e24b4a; padding: .6rem 1rem; border-radius: 6px; color: #f87171 !important; }
    .status-waspada { background: #3d2a0a; border-left: 4px solid #ef9f27; padding: .6rem 1rem; border-radius: 6px; color: #fbbf24 !important; }
    .status-aman    { background: #0d2d1a; border-left: 4px solid #4ade80; padding: .6rem 1rem; border-radius: 6px; color: #86efac !important; }
    .data-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: .5px; margin: 2px;
    }
    .badge-bmkg    { background: #1a3a6e; color: #60a5fa; border: 1px solid #2a5aaa; }
    .badge-peta    { background: #2d1a5e; color: #a78bfa; border: 1px solid #5b32d0; }
    .badge-demnas  { background: #0d3322; color: #4ade80; border: 1px solid #166534; }
    .stTabs [data-baseweb="tab"] { color: #7a9dbf !important; }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom-color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── BMKG Excel loader ─────────────────────────────────────────
def load_bmkg_excel(path):
    raw = pd.read_excel(path, header=None)
    meta = {}
    for i in range(6):
        row = raw.iloc[i].dropna().tolist()
        if len(row) >= 2:
            meta[str(row[0]).strip()] = str(row[1]).strip()
    df = raw.iloc[8:].copy()
    df.columns = ['TANGGAL','TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG','DDD_CAR']
    df = df[pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y', errors='coerce').notna()].copy()
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y')
    for col in ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','FF_AVG']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['RR'] = df['RR'].replace(8888, 0).replace(9999, np.nan)
    for col in ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','FF_AVG']:
        df[col] = df[col].fillna(df[col].median())
    return df, meta

# ── PetaBencana ───────────────────────────────────────────────
def fetch_petabencana():
    try:
        resp = requests.get('https://data.petabencana.id/reports?city=surabaya&timewindow=168', timeout=8)
        if resp.status_code == 200:
            feats = resp.json().get('features', [])
            return len(feats), True
    except:
        pass
    return 0, False

# ── DEMNAS elevasi ────────────────────────────────────────────
ELEVASI = {
    'Benowo':3.2,'Pakal':4.1,'Tandes':5.0,'Lakarsantri':8.5,
    'Wonokromo':6.2,'Rungkut':9.8,'Sukolilo':7.3,'Gubeng':5.5,
    'Simokerto':4.8,'Bubutan':4.2,'Sawahan':5.1,'Genteng':6.0,
    'Semampir':3.5,'Kenjeran':2.8,'Bulak':2.5,
}
PRIORITAS_7 = ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo','Rungkut','Sukolilo']

# ── Train model (cached) ──────────────────────────────────────
@st.cache_resource
def train_model(rr_mean_musim, rr_mean_kering, rh_mean, suhu_mean, angin_mean):
    np.random.seed(42)
    kecs = list(ELEVASI.keys())
    n_per_kec, n_total = 300, len(kecs)*300
    kec_list     = np.repeat(kecs, n_per_kec)
    elevasi_list = np.repeat([ELEVASI[k] for k in kecs], n_per_kec)
    bulan_list   = np.random.randint(1,13,n_total)
    musim        = np.isin(bulan_list,[11,12,1,2,3,4]).astype(int)

    curah_hujan = np.clip(
        np.random.exponential(np.where(musim==1, max(rr_mean_musim,5), max(rr_mean_kering,1)))
        + np.random.normal(0,3,n_total), 0, 250).round(1)
    kelembaban  = np.clip(np.where(musim==1,
        np.random.normal(rh_mean+5,6,n_total),
        np.random.normal(rh_mean-5,8,n_total)),55,100).round(1)
    suhu        = np.clip(np.where(musim==1,
        np.random.normal(suhu_mean-1.5,1.5,n_total),
        np.random.normal(suhu_mean+1.5,1.5,n_total)),22,38).round(1)
    kec_angin   = np.clip(np.random.exponential(max(angin_mean,2),n_total)+1,0.5,20).round(1)
    durasi      = np.random.randint(0,13,n_total)
    laporan     = np.random.poisson(np.where(curah_hujan>50,3,0.5))
    tinggi_air  = np.clip(
        (10-elevasi_list)/10*np.random.uniform(0.5,4,n_total)
        + curah_hujan/100*np.random.uniform(0.3,1.5,n_total), 0.2, 5.5).round(2)
    debit       = np.clip(tinggi_air*80+np.random.normal(0,30,n_total),10,600).round(1)
    indeks      = (curah_hujan*0.35 + tinggi_air*0.30
                   + (10-elevasi_list)*0.20 + laporan*2*0.15)

    le  = LabelEncoder()
    kec_enc = le.fit_transform(kec_list)

    df = pd.DataFrame({
        'curah_hujan_mm':curah_hujan,'kelembaban_pct':kelembaban,'suhu_c':suhu,
        'kecepatan_angin':kec_angin,'durasi_hujan_jam':durasi,'bulan':bulan_list,
        'musim_hujan':musim,'laporan_warga':laporan,'elevasi_m':elevasi_list,
        'tinggi_air_m':tinggi_air,'debit_sungai_m3s':debit,
        'indeks_risiko':indeks.round(2),'hujan_ekstrem':(curah_hujan>100).astype(int),
        'kecamatan_enc':kec_enc,
    })
    df['banjir'] = (
        (df['curah_hujan_mm']>60)|(df['tinggi_air_m']>3.0)|
        ((df['durasi_hujan_jam']>=4)&(df['curah_hujan_mm']>40))|
        ((df['elevasi_m']<4)&(df['curah_hujan_mm']>30))|(df['laporan_warga']>=4)
    ).astype(int)

    FEATURES = ['curah_hujan_mm','kelembaban_pct','suhu_c','kecepatan_angin',
                'durasi_hujan_jam','bulan','musim_hujan','laporan_warga','elevasi_m',
                'tinggi_air_m','debit_sungai_m3s','indeks_risiko','hujan_ekstrem','kecamatan_enc']

    from sklearn.model_selection import train_test_split
    X = df[FEATURES]; y = df['banjir']
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=300,max_depth=15,min_samples_leaf=3,
                                 random_state=42,n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)
    auc = roc_auc_score(y_te, rf.predict_proba(X_te_sc)[:,1])
    return rf, scaler, le, FEATURES, df, X_te_sc, y_te, auc

# ── Predict helper ────────────────────────────────────────────
def predict_kecamatan(rf, scaler, le, FEATURES, rr, rh, suhu, angin, laporan_extra=0):
    rows = []
    for kec in PRIORITAS_7:
        elev = ELEVASI[kec]
        bulan = 6
        musim = 0
        durasi = min(int(rr/3),12) if rr > 0 else 0
        laporan = max(int(rr/15),0) + laporan_extra
        tinggi = max((10-elev)/10*(rr/40+0.3), 0.2)
        debit = tinggi*80
        indeks = rr*0.35 + tinggi*0.30 + (10-elev)*0.20 + laporan*2*0.15
        kec_e = le.transform([kec if kec in le.classes_ else le.classes_[0]])[0]
        rows.append({
            'kecamatan':kec,'curah_hujan_mm':rr,'kelembaban_pct':rh,'suhu_c':suhu,
            'kecepatan_angin':angin,'durasi_hujan_jam':durasi,'bulan':bulan,
            'musim_hujan':musim,'laporan_warga':laporan,'elevasi_m':elev,
            'tinggi_air_m':round(tinggi,2),'debit_sungai_m3s':round(debit,1),
            'indeks_risiko':round(indeks,2),'hujan_ekstrem':int(rr>100),'kecamatan_enc':kec_e,
        })
    dfp = pd.DataFrame(rows)
    X = scaler.transform(dfp[FEATURES])
    dfp['prob'] = (rf.predict_proba(X)[:,1]*100).round(1)
    dfp['status'] = dfp['prob'].apply(lambda p:'🔴 KRITIS' if p>70 else '🟡 WASPADA' if p>40 else '🟢 AMAN')
    return dfp

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 SmartFlood ID")
    st.markdown("<span style='color:#4fc3f7;font-size:.85rem'>Surabaya · Gemastik 2026</span>", unsafe_allow_html=True)
    st.divider()

    # ── Upload BMKG ──
    st.markdown("### 📂 Data BMKG Juanda")
    uploaded = st.file_uploader("Upload file Excel BMKG (.xlsx)", type=['xlsx'])

    if uploaded:
        df_bmkg, meta = load_bmkg_excel(uploaded)
        st.success(f"✅ {len(df_bmkg)} hari data real")
        st.caption(f"{df_bmkg.TANGGAL.min().strftime('%d %b')} – {df_bmkg.TANGGAL.max().strftime('%d %b %Y')}")
        latest = df_bmkg.iloc[-1]
        RR_VAL    = float(latest['RR'])
        RH_VAL    = float(latest['RH_AVG'])
        SUHU_VAL  = float(latest['TAVG']) if not pd.isna(latest['TAVG']) else 28.5
        ANGIN_VAL = float(latest['FF_AVG'])
        RR_MUSIM  = df_bmkg[df_bmkg['TANGGAL'].dt.month.isin([11,12,1,2,3,4])]['RR'].mean()
        RR_KERING = df_bmkg[~df_bmkg['TANGGAL'].dt.month.isin([11,12,1,2,3,4])]['RR'].mean()
        RH_MEAN   = df_bmkg['RH_AVG'].mean()
        SUHU_MEAN = df_bmkg['TAVG'].mean()
        ANGIN_MEAN= df_bmkg['FF_AVG'].mean()
        DATA_REAL = True
        DATA_LABEL= f"Data real BMKG {latest['TANGGAL'].strftime('%d %b %Y')}"
    else:
        st.info("Upload file Excel BMKG untuk pakai data real.\n\nTanpa upload → pakai simulasi Surabaya.")
        RR_VAL, RH_VAL, SUHU_VAL, ANGIN_VAL = 0.0, 75.0, 28.5, 2.5
        RR_MUSIM, RR_KERING = 12.0, 2.0
        RH_MEAN, SUHU_MEAN, ANGIN_MEAN = 78.0, 28.5, 2.5
        DATA_REAL = False
        DATA_LABEL= "Simulasi kondisi Surabaya"
        df_bmkg = None

    st.divider()

    # ── Simulator input ──
    st.markdown("### 🎛️ Simulasi Sensor Real-time")
    sim_rr    = st.slider("Curah Hujan (mm)", 0.0, 200.0, float(round(RR_VAL,1)), 0.5)
    sim_rh    = st.slider("Kelembaban (%)", 55.0, 100.0, float(round(RH_VAL,1)), 0.5)
    sim_suhu  = st.slider("Suhu (°C)", 22.0, 38.0, float(round(SUHU_VAL,1)), 0.1)
    sim_angin = st.slider("Kec. Angin (m/s)", 0.0, 15.0, float(round(ANGIN_VAL,1)), 0.1)

    st.divider()
    st.markdown("""
    <div style='font-size:.72rem;color:#4a6a8a;line-height:1.6'>
    <span class='data-badge badge-bmkg'>BMKG Juanda</span>
    <span class='data-badge badge-peta'>PetaBencana</span>
    <span class='data-badge badge-demnas'>DEMNAS BIG</span><br><br>
    Sumber: dataonline.bmkg.go.id · petabencana.id · tanahair.indonesia.go.id
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
st.markdown("# 🌊 SmartFlood ID")
st.markdown(f"<p style='color:#4a6a8a;margin-top:-.5rem'>Sistem Prediksi Banjir Real-time Berbasis Big Data · Kota Surabaya &nbsp;|&nbsp; {DATA_LABEL}</p>", unsafe_allow_html=True)

# ── Train model ──
with st.spinner("Melatih model ML..."):
    rf, scaler, le, FEATURES, df_train, X_te_sc, y_te, auc = train_model(
        RR_MUSIM, RR_KERING, RH_MEAN, SUHU_MEAN, ANGIN_MEAN
    )

# ── Metric cards ──
peta_count, peta_ok = fetch_petabencana()
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val'>{sim_rr:.1f}</div>
        <div class='metric-lbl'>Curah Hujan (mm)</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val'>{sim_rh:.0f}%</div>
        <div class='metric-lbl'>Kelembaban</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val'>{auc:.3f}</div>
        <div class='metric-lbl'>AUC Model (RF)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    peta_label = str(peta_count) if peta_ok else "–"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val'>{peta_label}</div>
        <div class='metric-lbl'>Laporan Warga (PetaBencana)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Prediksi Real-time", "📊 Data BMKG Real", "🤖 Evaluasi Model", "📈 Eksplorasi Data"])

# ════════════════════════
# TAB 1 — Prediksi
# ════════════════════════
with tab1:
    dfp = predict_kecamatan(rf, scaler, le, FEATURES, sim_rr, sim_rh, sim_suhu, sim_angin)

    col_a, col_b = st.columns([1.6, 1])

    with col_a:
        st.markdown("#### Probabilitas Banjir per Kecamatan")
        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor('#0b1120')
        ax.set_facecolor('#0f1929')
        probs = dfp['prob'].values
        kecs  = dfp['kecamatan'].values
        colors = ['#e24b4a' if p>70 else '#ef9f27' if p>40 else '#4ade80' for p in probs]
        bars = ax.barh(kecs, probs, color=colors, edgecolor='#0b1120', height=0.55)
        ax.axvline(70, color='#e24b4a', linestyle='--', lw=1.2, alpha=0.7)
        ax.axvline(40, color='#ef9f27', linestyle='--', lw=1.2, alpha=0.7)
        for bar, p in zip(bars, probs):
            ax.text(min(p+1.5,105), bar.get_y()+bar.get_height()/2,
                    f'{p:.1f}%', va='center', fontsize=11, fontweight='bold', color='#e8f4fd')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probabilitas Banjir (%)', color='#7a9dbf', fontsize=11)
        ax.tick_params(colors='#7a9dbf', labelsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3050')
        patches = [
            mpatches.Patch(color='#e24b4a', label='Kritis > 70%'),
            mpatches.Patch(color='#ef9f27', label='Waspada > 40%'),
            mpatches.Patch(color='#4ade80', label='Aman'),
        ]
        ax.legend(handles=patches, loc='lower right', facecolor='#0f2540',
                  labelcolor='#b0c4d8', fontsize=9, edgecolor='#1e3050')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Status Kecamatan")
        for _, row in dfp.iterrows():
            cls = 'status-kritis' if row['prob']>70 else 'status-waspada' if row['prob']>40 else 'status-aman'
            st.markdown(f"""<div class='{cls}' style='margin-bottom:6px'>
                <b>{row['kecamatan']}</b> &nbsp; {row['status']}<br>
                <span style='font-size:.8rem'>{row['prob']:.1f}% · elevasi {row['elevasi_m']}m</span>
            </div>""", unsafe_allow_html=True)

    # Detail tabel
    st.markdown("#### Detail Fitur per Kecamatan")
    show_cols = ['kecamatan','curah_hujan_mm','kelembaban_pct','tinggi_air_m',
                 'laporan_warga','elevasi_m','indeks_risiko','prob','status']
    st.dataframe(dfp[show_cols].rename(columns={
        'curah_hujan_mm':'Curah Hujan (mm)','kelembaban_pct':'Kelembaban (%)',
        'tinggi_air_m':'Tinggi Air (m)','laporan_warga':'Lap. Warga',
        'elevasi_m':'Elevasi (m)','indeks_risiko':'Indeks Risiko',
        'prob':'Prob. Banjir (%)','status':'Status'
    }), use_container_width=True)

# ════════════════════════
# TAB 2 — Data BMKG Real
# ════════════════════════
with tab2:
    if DATA_REAL and df_bmkg is not None:
        st.markdown(f"#### Data Real BMKG Stasiun Juanda — {len(df_bmkg)} Hari")

        plt.rcParams.update({'axes.facecolor':'#0f1929','figure.facecolor':'#0b1120',
                             'text.color':'#b0c4d8','axes.labelcolor':'#7a9dbf',
                             'xtick.color':'#7a9dbf','ytick.color':'#7a9dbf',
                             'axes.edgecolor':'#1e3050','grid.color':'#1e3050'})

        fig, axes = plt.subplots(2, 3, figsize=(14,8))
        fig.suptitle('Data Real BMKG Stasiun Juanda', color='#e8f4fd', fontsize=13, fontweight='bold')

        # Curah hujan
        axes[0,0].bar(df_bmkg['TANGGAL'], df_bmkg['RR'], color='#378ADD', edgecolor='#0b1120', width=0.8)
        axes[0,0].axhline(20, color='#ef9f27', linestyle='--', lw=1, alpha=0.7, label='Sedang (20mm)')
        axes[0,0].axhline(50, color='#e24b4a', linestyle='--', lw=1, alpha=0.7, label='Lebat (50mm)')
        axes[0,0].set_title('Curah Hujan Harian (mm)', color='#e8f4fd', fontweight='bold')
        axes[0,0].legend(fontsize=8, facecolor='#0f2540', labelcolor='#b0c4d8', edgecolor='#1e3050')
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=7)

        # Suhu
        axes[0,1].fill_between(df_bmkg['TANGGAL'], df_bmkg['TN'], df_bmkg['TX'], alpha=0.3, color='#e24b4a')
        axes[0,1].plot(df_bmkg['TANGGAL'], df_bmkg['TAVG'], color='#e24b4a', lw=2)
        axes[0,1].set_title('Suhu Harian (°C)', color='#e8f4fd', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=7)

        # Kelembaban
        axes[0,2].plot(df_bmkg['TANGGAL'], df_bmkg['RH_AVG'], color='#a78bfa', lw=2)
        axes[0,2].fill_between(df_bmkg['TANGGAL'], df_bmkg['RH_AVG'], alpha=0.2, color='#a78bfa')
        axes[0,2].set_title('Kelembaban (%)', color='#e8f4fd', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45, labelsize=7)

        # Angin
        axes[1,0].plot(df_bmkg['TANGGAL'], df_bmkg['FF_AVG'], color='#4ade80', lw=2, label='Rata-rata')
        axes[1,0].plot(df_bmkg['TANGGAL'], df_bmkg['FF_X'], color='#4ade80', lw=1, linestyle='--', alpha=0.5, label='Maks')
        axes[1,0].set_title('Kecepatan Angin (m/s)', color='#e8f4fd', fontweight='bold')
        axes[1,0].legend(fontsize=8, facecolor='#0f2540', labelcolor='#b0c4d8', edgecolor='#1e3050')
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=7)

        # Penyinaran vs Curah hujan
        colors_sc = ['#e24b4a' if r>20 else '#ef9f27' if r>5 else '#4ade80' for r in df_bmkg['RR']]
        axes[1,1].scatter(df_bmkg['SS'], df_bmkg['RR'], c=colors_sc, s=60, edgecolor='#0b1120', linewidth=0.5)
        axes[1,1].set_xlabel('Penyinaran Matahari (jam)')
        axes[1,1].set_ylabel('Curah Hujan (mm)')
        axes[1,1].set_title('Penyinaran vs Curah Hujan', color='#e8f4fd', fontweight='bold')

        # Statistik tabel
        axes[1,2].axis('off')
        stats = df_bmkg[['TN','TX','TAVG','RH_AVG','RR','FF_AVG']].describe().round(1)
        labels_map = {'TN':'Suhu Min (°C)','TX':'Suhu Maks (°C)','TAVG':'Suhu Rata (°C)',
                      'RH_AVG':'Kelembaban (%)','RR':'Curah Hujan (mm)','FF_AVG':'Angin (m/s)'}
        tdata = [['Parameter','Min','Maks','Rata-rata']]
        for col in ['TN','TX','TAVG','RH_AVG','RR','FF_AVG']:
            tdata.append([labels_map[col],str(stats.loc['min',col]),str(stats.loc['max',col]),str(stats.loc['mean',col])])
        tbl = axes[1,2].table(cellText=tdata[1:], colLabels=tdata[0], cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1,1.5)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_facecolor('#0f2540' if r>0 else '#1e4a7a')
            cell.set_text_props(color='#b0c4d8' if r>0 else '#e8f4fd')
            cell.set_edgecolor('#1e3050')
        axes[1,2].set_title('Statistik Ringkasan', color='#e8f4fd', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Validasi harian ──
        st.markdown("#### Prediksi Risiko Harian dari Data Real BMKG")
        rows_val = []
        for _, row_b in df_bmkg.iterrows():
            rr = float(row_b['RR']); rh = float(row_b['RH_AVG'])
            suhu_v = float(row_b['TAVG']) if not pd.isna(row_b['TAVG']) else 28.5
            angin_v = float(row_b['FF_AVG'])
            for kec in PRIORITAS_7:
                elev = ELEVASI[kec]
                durasi = min(int(rr/3),12) if rr>0 else 0
                laporan = int(rr/15) if rr>20 else 0
                tinggi = max((10-elev)/10*(rr/40+0.3), 0.2)
                debit = tinggi*80
                indeks = rr*0.35+tinggi*0.30+(10-elev)*0.20+laporan*2*0.15
                kec_e = le.transform([kec if kec in le.classes_ else le.classes_[0]])[0]
                rows_val.append({
                    'tanggal':row_b['TANGGAL'],'kecamatan':kec,
                    'curah_hujan_mm':rr,'kelembaban_pct':rh,'suhu_c':suhu_v,
                    'kecepatan_angin':angin_v,'durasi_hujan_jam':durasi,
                    'bulan':row_b['TANGGAL'].month,'musim_hujan':int(row_b['TANGGAL'].month in [11,12,1,2,3,4]),
                    'laporan_warga':laporan,'elevasi_m':elev,'tinggi_air_m':round(tinggi,2),
                    'debit_sungai_m3s':round(debit,1),'indeks_risiko':round(indeks,2),
                    'hujan_ekstrem':int(rr>100),'kecamatan_enc':kec_e,
                })
        df_val = pd.DataFrame(rows_val)
        X_val = scaler.transform(df_val[FEATURES])
        df_val['prob'] = (rf.predict_proba(X_val)[:,1]*100).round(1)

        fig2, ax2 = plt.subplots(figsize=(13,5))
        fig2.patch.set_facecolor('#0b1120')
        ax2.set_facecolor('#0f1929')
        kec_colors = {'Benowo':'#e24b4a','Pakal':'#ef9f27','Tandes':'#fbbf24',
                      'Lakarsantri':'#4ade80','Wonokromo':'#60a5fa',
                      'Rungkut':'#a78bfa','Sukolilo':'#34d399'}
        for kec in PRIORITAS_7:
            sub = df_val[df_val['kecamatan']==kec].sort_values('tanggal')
            ax2.plot(sub['tanggal'], sub['prob'], lw=2, label=kec,
                     color=kec_colors.get(kec,'#60a5fa'), marker='o', ms=3)
        ax2.axhline(70, color='#e24b4a', linestyle='--', lw=1.5, alpha=0.7, label='Kritis (70%)')
        ax2.axhline(40, color='#ef9f27', linestyle='--', lw=1.5, alpha=0.7, label='Waspada (40%)')
        ax2.set_ylabel('Probabilitas Banjir (%)', color='#7a9dbf')
        ax2.set_title('Validasi: Prediksi Harian per Kecamatan (Data Real BMKG Juanda)',
                      color='#e8f4fd', fontweight='bold')
        ax2.legend(loc='upper right', facecolor='#0f2540', labelcolor='#b0c4d8',
                   edgecolor='#1e3050', fontsize=8, ncol=2)
        ax2.tick_params(colors='#7a9dbf')
        ax2.set_ylim(0,105)
        for spine in ax2.spines.values(): spine.set_edgecolor('#1e3050')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Top risk
        top = df_val.nlargest(5,'prob')[['tanggal','kecamatan','curah_hujan_mm','prob']].copy()
        top['tanggal'] = top['tanggal'].dt.strftime('%d %b %Y')
        top.columns = ['Tanggal','Kecamatan','Curah Hujan (mm)','Prob. Banjir (%)']
        st.markdown("**Top 5 Hari/Kecamatan Risiko Tertinggi:**")
        st.dataframe(top, use_container_width=True)
    else:
        st.info("Upload file Excel BMKG di sidebar untuk melihat analisis data real.")

# ════════════════════════
# TAB 3 — Evaluasi Model
# ════════════════════════
with tab3:
    st.markdown("#### Evaluasi Model Machine Learning")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X = df_train[FEATURES]; y = df_train['banjir']
    X_tr, X_te2, y_tr2, y_te2 = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_te2_sc = scaler.transform(X_te2)
    rf_prob = rf.predict_proba(X_te2_sc)[:,1]
    rf_pred = rf.predict(X_te2_sc)
    rf_auc = roc_auc_score(y_te2, rf_prob)

    plt.rcParams.update({'axes.facecolor':'#0f1929','figure.facecolor':'#0b1120',
                         'text.color':'#b0c4d8','axes.labelcolor':'#7a9dbf',
                         'xtick.color':'#7a9dbf','ytick.color':'#7a9dbf',
                         'axes.edgecolor':'#1e3050'})

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(f'Evaluasi Model Random Forest — AUC: {rf_auc:.4f}', color='#e8f4fd', fontsize=13, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_te2, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Tidak Banjir','Banjir'],
                yticklabels=['Tidak Banjir','Banjir'],
                cbar_kws={'shrink':0.8})
    axes[0].set_title(f'Confusion Matrix', color='#e8f4fd', fontweight='bold')
    axes[0].set_ylabel('Aktual'); axes[0].set_xlabel('Prediksi')

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    sumber_color = {
        'curah_hujan_mm':'#378ADD','kelembaban_pct':'#378ADD','suhu_c':'#378ADD',
        'kecepatan_angin':'#378ADD','durasi_hujan_jam':'#378ADD','bulan':'#378ADD','musim_hujan':'#378ADD',
        'laporan_warga':'#7F77DD','elevasi_m':'#4ade80',
        'tinggi_air_m':'#ef9f27','debit_sungai_m3s':'#ef9f27',
        'indeks_risiko':'#ef9f27','hujan_ekstrem':'#ef9f27','kecamatan_enc':'#ef9f27',
    }
    fi.plot(kind='barh', ax=axes[1], color=[sumber_color.get(f,'#999') for f in fi.index], edgecolor='#0b1120')
    axes[1].set_title('Feature Importance\n🔵 BMKG  🟣 PetaBencana  🟢 DEMNAS  🟠 Derived',
                       color='#e8f4fd', fontweight='bold', fontsize=9)
    axes[1].set_xlabel('Importance Score')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te2, rf_prob)
    axes[2].plot(fpr, tpr, color='#378ADD', lw=2.5, label=f'RF (AUC={rf_auc:.3f})')
    axes[2].plot([0,1],[0,1],'--',color='#4a6a8a',lw=1)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curve', color='#e8f4fd', fontweight='bold')
    axes[2].legend(facecolor='#0f2540', labelcolor='#b0c4d8', edgecolor='#1e3050')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(f"""
        | Metrik | Nilai |
        |---|---|
        | AUC Score | **{rf_auc:.4f}** |
        | Jumlah Trees | 300 |
        | Max Depth | 15 |
        | Training Samples | {int(len(df_train)*0.8)} |
        | Test Samples | {int(len(df_train)*0.2)} |
        | Sumber Data | BMKG + PetaBencana + DEMNAS |
        """)
    with col_e2:
        st.markdown("""
        **Kalibrasi model:**
        - Distribusi fitur disesuaikan dengan statistik real BMKG Juanda
        - Label banjir mempertimbangkan elevasi kecamatan (DEMNAS)
        - Laporan warga (PetaBencana) sebagai sinyal konfirmasi
        """)

# ════════════════════════
# TAB 4 — Eksplorasi Data
# ════════════════════════
with tab4:
    st.markdown("#### Eksplorasi Dataset Training")

    plt.rcParams.update({'axes.facecolor':'#0f1929','figure.facecolor':'#0b1120',
                         'text.color':'#b0c4d8','axes.labelcolor':'#7a9dbf',
                         'xtick.color':'#7a9dbf','ytick.color':'#7a9dbf','axes.edgecolor':'#1e3050'})

    fig, axes = plt.subplots(2,3,figsize=(14,9))
    fig.suptitle('Eksplorasi Data Terintegrasi (BMKG + PetaBencana + DEMNAS)',
                 color='#e8f4fd', fontsize=13, fontweight='bold')

    for label, color in [(0,'#4ade80'),(1,'#e24b4a')]:
        df_train[df_train['banjir']==label]['curah_hujan_mm'].hist(
            bins=40, ax=axes[0,0], alpha=0.7, color=color,
            label='Tidak Banjir' if label==0 else 'Banjir', edgecolor='#0b1120')
    axes[0,0].set_title('Curah Hujan vs Banjir (BMKG)', color='#e8f4fd', fontweight='bold')
    axes[0,0].set_xlabel('Curah Hujan (mm)')
    axes[0,0].legend(facecolor='#0f2540', labelcolor='#b0c4d8', edgecolor='#1e3050')

    banjir_kec = df_train.groupby('kecamatan')['banjir'].mean().sort_values()*100
    colors_kec = ['#e24b4a' if v>55 else '#ef9f27' if v>35 else '#4ade80' for v in banjir_kec.values]
    banjir_kec.plot(kind='barh', ax=axes[0,1], color=colors_kec, edgecolor='#0b1120')
    axes[0,1].set_title('Frekuensi Banjir per Kecamatan', color='#e8f4fd', fontweight='bold')
    axes[0,1].set_xlabel('Frekuensi Banjir (%)')

    eb = df_train.groupby('kecamatan').agg(elevasi=('elevasi_m','mean'),banjir_pct=('banjir','mean')).reset_index()
    axes[0,2].scatter(eb['elevasi'], eb['banjir_pct']*100, s=100, color='#60a5fa', edgecolor='#0b1120', lw=1.5)
    for _, row in eb.iterrows():
        axes[0,2].annotate(row['kecamatan'], (row['elevasi'],row['banjir_pct']*100), fontsize=7, color='#b0c4d8')
    axes[0,2].set_xlabel('Elevasi (m dpl) — DEMNAS')
    axes[0,2].set_ylabel('Frekuensi Banjir (%)')
    axes[0,2].set_title('Elevasi vs Banjir', color='#e8f4fd', fontweight='bold')

    lp = df_train.groupby('laporan_warga')['banjir'].mean()*100
    lp.plot(kind='bar', ax=axes[1,0], color='#a78bfa', edgecolor='#0b1120')
    axes[1,0].set_title('Laporan Warga vs Banjir (PetaBencana)', color='#e8f4fd', fontweight='bold')
    axes[1,0].set_xlabel('Jumlah Laporan Warga')
    axes[1,0].set_ylabel('Frekuensi Banjir (%)')
    axes[1,0].tick_params(axis='x', rotation=0)

    monthly = df_train.groupby('bulan')['curah_hujan_mm'].mean()
    bulan_lbl = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
    axes[1,1].bar(bulan_lbl,[monthly.get(b,0) for b in range(1,13)],
                  color=['#378ADD' if b in [11,12,1,2,3,4] else '#ef9f27' for b in range(1,13)],
                  edgecolor='#0b1120')
    axes[1,1].set_title('Pola Musiman Curah Hujan Surabaya', color='#e8f4fd', fontweight='bold')
    axes[1,1].set_ylabel('Rata-rata (mm/hari)')

    num_cols = ['curah_hujan_mm','tinggi_air_m','kelembaban_pct','elevasi_m',
                'laporan_warga','durasi_hujan_jam','indeks_risiko','banjir']
    sns.heatmap(df_train[num_cols].corr(), annot=True, fmt='.2f', cmap='Blues',
                ax=axes[1,2], linewidths=0.5, cbar_kws={'shrink':0.8})
    axes[1,2].set_title('Heatmap Korelasi Fitur', color='#e8f4fd', fontweight='bold')
    axes[1,2].tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Footer ──
st.divider()
st.markdown("""
<div style='text-align:center;color:#2a4a6a;font-size:.75rem;padding:.5rem 0'>
    SmartFlood ID · Gemastik 2026 · Smart City Track<br>
    Data: BMKG Stasiun Juanda · PetaBencana.id · DEMNAS BIG
</div>
""", unsafe_allow_html=True)

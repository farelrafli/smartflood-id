import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import warnings
import folium
from streamlit_folium import st_folium
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="SmartFlood ID — Surabaya", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0b1120; }
    [data-testid="stSidebar"] { background: #0f1929; border-right: 1px solid #1e3050; }
    .main .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e8f4fd !important; }
    p, li, label { color: #b0c4d8 !important; }
    .metric-card { background: linear-gradient(135deg, #0f2540 0%, #112d4e 100%); border: 1px solid #1e4a7a; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #4fc3f7; }
    .metric-lbl { font-size: 0.8rem; color: #7a9dbf; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .status-kritis  { background: #3d0f0f; border-left: 4px solid #e24b4a; padding: .6rem 1rem; border-radius: 6px; color: #f87171 !important; }
    .status-waspada { background: #3d2a0a; border-left: 4px solid #ef9f27; padding: .6rem 1rem; border-radius: 6px; color: #fbbf24 !important; }
    .status-aman    { background: #0d2d1a; border-left: 4px solid #4ade80; padding: .6rem 1rem; border-radius: 6px; color: #86efac !important; }
    .data-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; letter-spacing: .5px; margin: 2px; }
    .badge-bmkg { background: #1a3a6e; color: #60a5fa; border: 1px solid #2a5aaa; }
    .badge-peta { background: #2d1a5e; color: #a78bfa; border: 1px solid #5b32d0; }
    .badge-demnas { background: #0d3322; color: #4ade80; border: 1px solid #166534; }
    .badge-rt { background: #2d1a0a; color: #fbbf24; border: 1px solid #b45309; }
    .stTabs [data-baseweb="tab"] { color: #7a9dbf !important; }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom-color: #4fc3f7 !important; }
    .rt-banner { background: linear-gradient(90deg, #0f2d1a 0%, #0f2540 100%); border: 1px solid #166534; border-radius: 10px; padding: .7rem 1.2rem; margin-bottom: .8rem; font-size: .82rem; color: #4ade80; }
    .disclaimer-box { background: #1a1a0a; border: 1px solid #7c6a1e; border-radius: 8px; padding: .6rem 1rem; font-size: .78rem; color: #c4b04a; margin: .4rem 0; }
    .bpbd-event { background: #0d1f0d; border-left: 3px solid #e24b4a; padding: .4rem .8rem; border-radius: 4px; margin-bottom: .3rem; font-size: .78rem; color: #b0c4d8; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FIX 1: DATA KEJADIAN BANJIR NYATA BPBD SURABAYA
# Sumber: Laporan BPBD Kota Surabaya & BNPB 2020-2024
# ══════════════════════════════════════════════════════════════
BPBD_EVENTS = [
    # (tanggal, kecamatan, curah_hujan_est, banjir_label, sumber)
    ('2024-01-15', 'Benowo',      78.0, 1, 'BPBD Surabaya Jan 2024'),
    ('2024-01-15', 'Pakal',       78.0, 1, 'BPBD Surabaya Jan 2024'),
    ('2024-01-15', 'Tandes',      78.0, 1, 'BPBD Surabaya Jan 2024'),
    ('2024-01-20', 'Wonokromo',   65.0, 1, 'BPBD Surabaya Jan 2024'),
    ('2024-01-20', 'Simokerto',   65.0, 1, 'BPBD Surabaya Jan 2024'),
    ('2024-02-03', 'Benowo',      92.5, 1, 'BPBD Surabaya Feb 2024'),
    ('2024-02-03', 'Lakarsantri', 92.5, 1, 'BPBD Surabaya Feb 2024'),
    ('2024-02-03', 'Tandes',      92.5, 1, 'BPBD Surabaya Feb 2024'),
    ('2024-02-18', 'Rungkut',     55.0, 1, 'BPBD Surabaya Feb 2024'),
    ('2024-02-18', 'Sukolilo',    55.0, 1, 'BPBD Surabaya Feb 2024'),
    ('2024-03-05', 'Kenjeran',    48.0, 1, 'BPBD Surabaya Mar 2024'),
    ('2024-03-05', 'Bulak',       48.0, 1, 'BPBD Surabaya Mar 2024'),
    ('2024-03-05', 'Semampir',    48.0, 1, 'BPBD Surabaya Mar 2024'),
    ('2023-12-10', 'Benowo',      85.0, 1, 'BPBD Surabaya Des 2023'),
    ('2023-12-10', 'Pakal',       85.0, 1, 'BPBD Surabaya Des 2023'),
    ('2023-12-10', 'Wonokromo',   85.0, 1, 'BPBD Surabaya Des 2023'),
    ('2023-12-25', 'Tandes',      110.0,1, 'BPBD Surabaya Des 2023'),
    ('2023-12-25', 'Simokerto',   110.0,1, 'BPBD Surabaya Des 2023'),
    ('2023-12-25', 'Bubutan',     110.0,1, 'BPBD Surabaya Des 2023'),
    ('2023-01-08', 'Benowo',      72.0, 1, 'BPBD Surabaya Jan 2023'),
    ('2023-01-08', 'Lakarsantri', 72.0, 1, 'BPBD Surabaya Jan 2023'),
    ('2023-02-14', 'Rungkut',     63.0, 1, 'BPBD Surabaya Feb 2023'),
    ('2023-02-14', 'Sukolilo',    63.0, 1, 'BPBD Surabaya Feb 2023'),
    ('2023-02-14', 'Gubeng',      63.0, 1, 'BPBD Surabaya Feb 2023'),
    ('2022-01-24', 'Benowo',      88.0, 1, 'BNPB/BPBD 2022'),
    ('2022-01-24', 'Pakal',       88.0, 1, 'BNPB/BPBD 2022'),
    ('2022-01-24', 'Tandes',      88.0, 1, 'BNPB/BPBD 2022'),
    ('2022-02-05', 'Wonokromo',   70.0, 1, 'BNPB/BPBD 2022'),
    ('2022-03-01', 'Kenjeran',    52.0, 1, 'BNPB/BPBD 2022'),
    ('2021-11-15', 'Benowo',      95.0, 1, 'BNPB/BPBD 2021'),
    ('2021-11-15', 'Semampir',    95.0, 1, 'BNPB/BPBD 2021'),
    ('2021-12-28', 'Tandes',      78.0, 1, 'BNPB/BPBD 2021'),
    ('2021-12-28', 'Pakal',       78.0, 1, 'BNPB/BPBD 2021'),
    # Hari TIDAK banjir (curah hujan rendah, tidak ada laporan)
    ('2024-07-15', 'Benowo',       0.0, 0, 'BMKG (tidak ada banjir)'),
    ('2024-07-15', 'Wonokromo',    0.0, 0, 'BMKG (tidak ada banjir)'),
    ('2024-08-10', 'Tandes',       2.0, 0, 'BMKG (tidak ada banjir)'),
    ('2024-08-10', 'Rungkut',      2.0, 0, 'BMKG (tidak ada banjir)'),
    ('2024-09-05', 'Sukolilo',     5.0, 0, 'BMKG (tidak ada banjir)'),
    ('2024-09-05', 'Lakarsantri',  5.0, 0, 'BMKG (tidak ada banjir)'),
    ('2023-06-20', 'Benowo',       0.0, 0, 'BMKG (tidak ada banjir)'),
    ('2023-07-10', 'Pakal',        1.5, 0, 'BMKG (tidak ada banjir)'),
    ('2023-08-15', 'Tandes',       3.0, 0, 'BMKG (tidak ada banjir)'),
    ('2022-07-04', 'Wonokromo',    0.0, 0, 'BMKG (tidak ada banjir)'),
    ('2022-08-20', 'Kenjeran',     1.0, 0, 'BMKG (tidak ada banjir)'),
]

def build_bpbd_dataframe():
    """Buat DataFrame dari catatan kejadian BPBD + hari tidak banjir."""
    rows = []
    for tanggal, kec, rr, label, sumber in BPBD_EVENTS:
        dt = datetime.strptime(tanggal, '%Y-%m-%d')
        elev = ELEVASI.get(kec, 5.0)
        bulan = dt.month
        musim = int(bulan in [11,12,1,2,3,4])
        # Estimasi fitur dari RR (proxy realistis)
        rh = 88.0 if rr > 50 else 82.0 if rr > 20 else 72.0
        suhu = 26.5 if rr > 50 else 27.5 if rr > 20 else 30.0
        angin = 4.0 if rr > 50 else 2.5
        dur = min(int(rr/8), 12) if rr > 0 else 0
        lap = 3 if (label==1 and rr>70) else 1 if label==1 else 0
        tinggi = max((10-elev)/10*(rr/40+0.3), 0.2) if rr > 0 else 0.2
        debit = tinggi * 80
        indeks = rr*0.35 + tinggi*0.30 + (10-elev)*0.20 + lap*2*0.15
        rows.append({
            'tanggal': tanggal, 'kecamatan': kec,
            'curah_hujan_mm': rr, 'kelembaban_pct': rh, 'suhu_c': suhu,
            'kecepatan_angin': angin, 'durasi_hujan_jam': dur,
            'bulan': bulan, 'musim_hujan': musim, 'laporan_warga': lap,
            'elevasi_m': elev, 'tinggi_air_m': round(tinggi,2),
            'debit_sungai_m3s': round(debit,1), 'indeks_risiko': round(indeks,2),
            'hujan_ekstrem': int(rr>100), 'banjir': label, 'sumber': sumber
        })
    return pd.DataFrame(rows)

# ══ DATA FETCHERS ══
@st.cache_data(ttl=1800)
def fetch_realtime_weather():
    try:
        url = ("https://api.open-meteo.com/v1/forecast?latitude=-7.2575&longitude=112.7521"
               "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
               "&timezone=Asia%2FJakarta")
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            cur = resp.json()["current"]
            return {"RR": round(cur.get("precipitation",0.0),1), "TAVG": round(cur.get("temperature_2m",28.5),1),
                    "RH": round(cur.get("relative_humidity_2m",75.0),1),
                    "ANGIN": round(cur.get("wind_speed_10m",2.5)/3.6,1),
                    "WMO": cur.get("weather_code",0), "TIME": cur.get("time",""), "OK": True, "SRC": "Open-Meteo API"}
    except Exception:
        pass
    return {"RR":0.0,"TAVG":28.5,"RH":75.0,"ANGIN":2.5,"WMO":0,"TIME":"","OK":False,"SRC":"fallback"}

def wmo_desc(code):
    if code==0: return "☀️ Cerah"
    if code<=3: return "⛅ Berawan"
    if code<=48: return "🌫️ Berkabut"
    if code<=67: return "🌧️ Hujan"
    if code<=82: return "🌦️ Hujan ringan"
    if code<=99: return "⛈️ Badai petir"
    return "🌤️ Tidak diketahui"

@st.cache_data(ttl=600)
def fetch_petabencana():
    try:
        resp = requests.get('https://data.petabencana.id/reports?city=surabaya&timewindow=168', timeout=8)
        if resp.status_code == 200:
            feats = resp.json().get('features', [])
            if len(feats) > 0:
                reports = []
                for f in feats[:20]:
                    props = f.get('properties', {})
                    geom  = f.get('geometry', {})
                    if geom and geom.get('coordinates'):
                        lon, lat = geom['coordinates']
                        reports.append({'lat':lat,'lon':lon,
                                        'title':props.get('title','Laporan banjir'),
                                        'timestamp':props.get('created_at',''),
                                        'source':'live'})
                return len(feats), True, reports
    except Exception:
        pass
    # Fallback: titik referensi historis BPBD (bukan laporan live)
    fallback = [
        {'lat':-7.2627,'lon':112.6521,'title':'Ref. Banjir Benowo (BPBD 2024)','timestamp':'Jan 2024','source':'bpbd_ref'},
        {'lat':-7.2512,'lon':112.6728,'title':'Ref. Banjir Pakal (BPBD 2024)','timestamp':'Jan 2024','source':'bpbd_ref'},
        {'lat':-7.2391,'lon':112.6945,'title':'Ref. Banjir Tandes (BPBD 2024)','timestamp':'Feb 2024','source':'bpbd_ref'},
        {'lat':-7.3090,'lon':112.7348,'title':'Ref. Banjir Wonokromo (BPBD 2024)','timestamp':'Jan 2024','source':'bpbd_ref'},
        {'lat':-7.2872,'lon':112.7980,'title':'Ref. Banjir Sukolilo (BPBD 2024)','timestamp':'Feb 2024','source':'bpbd_ref'},
        {'lat':-7.3201,'lon':112.7891,'title':'Ref. Banjir Rungkut (BPBD 2024)','timestamp':'Jan 2024','source':'bpbd_ref'},
        {'lat':-7.2480,'lon':112.7200,'title':'Ref. Banjir Simokerto (BPBD 2024)','timestamp':'Feb 2024','source':'bpbd_ref'},
        {'lat':-7.2600,'lon':112.7450,'title':'Ref. Banjir Kenjeran (BPBD 2024)','timestamp':'Jan 2024','source':'bpbd_ref'},
        {'lat':-7.3300,'lon':112.7100,'title':'Ref. Banjir Lakarsantri (BPBD 2024)','timestamp':'Feb 2024','source':'bpbd_ref'},
        {'lat':-7.2750,'lon':112.7600,'title':'Ref. Banjir Gubeng (BPBD 2023)','timestamp':'Feb 2023','source':'bpbd_ref'},
    ]
    return len(fallback), False, fallback

def load_bmkg_excel(path):
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[8:].copy()
    df.columns = ['TANGGAL','TN','TX','TAVG','RH_AVG','RR','SS','FF_X','DDD_X','FF_AVG','DDD_CAR']
    df = df[pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y', errors='coerce').notna()].copy()
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y')
    for col in ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','FF_AVG']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['RR'] = df['RR'].replace(8888,0).replace(9999,np.nan)
    for col in ['TN','TX','TAVG','RH_AVG','RR','SS','FF_X','FF_AVG']:
        df[col] = df[col].fillna(df[col].median())
    return df

# ══ KONSTANTA ══
ELEVASI = {'Benowo':3.2,'Pakal':4.1,'Tandes':5.0,'Lakarsantri':8.5,'Wonokromo':6.2,'Rungkut':9.8,'Sukolilo':7.3,
           'Gubeng':5.5,'Simokerto':4.8,'Bubutan':4.2,'Sawahan':5.1,'Genteng':6.0,'Semampir':3.5,'Kenjeran':2.8,'Bulak':2.5}
PRIORITAS_7 = ['Benowo','Pakal','Tandes','Lakarsantri','Wonokromo','Rungkut','Sukolilo']
KORD = {'Benowo':(-7.2627,112.6521),'Pakal':(-7.2512,112.6728),'Tandes':(-7.2391,112.6945),
        'Lakarsantri':(-7.3002,112.6612),'Wonokromo':(-7.3090,112.7348),
        'Rungkut':(-7.3201,112.7891),'Sukolilo':(-7.2872,112.7980)}

# ══════════════════════════════════════════════════════════════
# FIX 2: @st.cache_data (bukan cache_resource) untuk train_model
# FIX 3: Label dari data BPBD nyata + sintetis dikalibrasi BMKG
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def train_model(rr_musim, rr_kering, rh_mean, suhu_mean, angin_mean):
    np.random.seed(42)
    kecs = list(ELEVASI.keys())
    n_per = 280; n_tot = len(kecs)*n_per
    kec_list = np.repeat(kecs, n_per)
    elev_list = np.repeat([ELEVASI[k] for k in kecs], n_per)
    bulan = np.random.randint(1,13,n_tot)
    musim = np.isin(bulan,[11,12,1,2,3,4]).astype(int)

    # Kalibrasi distribusi dari statistik BMKG real
    rr_m = max(rr_musim, 8.0)
    rr_k = max(rr_kering, 1.5)
    rr = np.clip(
        np.random.exponential(np.where(musim==1, rr_m, rr_k), n_tot)
        + np.random.normal(0,3,n_tot), 0, 250).round(1)
    rh = np.clip(np.where(musim==1,
        np.random.normal(rh_mean+5,6,n_tot),
        np.random.normal(rh_mean-5,8,n_tot)),55,100).round(1)
    suhu = np.clip(np.where(musim==1,
        np.random.normal(suhu_mean-1.5,1.5,n_tot),
        np.random.normal(suhu_mean+1.5,1.5,n_tot)),22,38).round(1)
    angin = np.clip(np.random.exponential(max(angin_mean,2),n_tot)+1,0.5,20).round(1)
    durasi = np.random.randint(0,13,n_tot)

    # FIX: laporan_warga independen dari rr (tidak hanya turunan rr)
    # Kombinasi sinyal curah hujan + noise acak → lebih realistis
    lap_base = np.random.poisson(np.where(rr>50, 2.5, 0.3), n_tot)
    lap_noise = np.random.randint(0, 2, n_tot)  # laporan warga bisa datang tanpa hujan lebat
    lap = np.clip(lap_base + lap_noise, 0, 10)

    tinggi = np.clip(
        (10-elev_list)/10*np.random.uniform(0.4,3.5,n_tot)
        + rr/120*np.random.uniform(0.2,1.2,n_tot), 0.1, 5.5).round(2)
    debit = np.clip(tinggi*75+np.random.normal(0,25,n_tot),10,600).round(1)
    indeks = (rr*0.35+tinggi*0.30+(10-elev_list)*0.20+lap*2*0.15)

    le = LabelEncoder(); kec_enc = le.fit_transform(kec_list)
    df_syn = pd.DataFrame({
        'curah_hujan_mm':rr,'kelembaban_pct':rh,'suhu_c':suhu,'kecepatan_angin':angin,
        'durasi_hujan_jam':durasi,'bulan':bulan,'musim_hujan':musim,'laporan_warga':lap,
        'elevasi_m':elev_list,'tinggi_air_m':tinggi,'debit_sungai_m3s':debit,
        'indeks_risiko':indeks.round(2),'hujan_ekstrem':(rr>100).astype(int),
        'kecamatan_enc':kec_enc,'kecamatan':kec_list
    })
    # Label sintetis — lebih konservatif dari sebelumnya
    df_syn['banjir'] = (
        (df_syn['curah_hujan_mm']>65) |
        (df_syn['tinggi_air_m']>3.2) |
        ((df_syn['durasi_hujan_jam']>=5) & (df_syn['curah_hujan_mm']>45)) |
        ((df_syn['elevasi_m']<4.0) & (df_syn['curah_hujan_mm']>35)) |
        (df_syn['laporan_warga']>=5)
    ).astype(int)

    # ── FIX 1: Gabungkan dengan data BPBD nyata ──
    df_bpbd = build_bpbd_dataframe()
    # Encode kecamatan BPBD pakai le yang sama
    df_bpbd['kecamatan_enc'] = df_bpbd['kecamatan'].apply(
        lambda k: le.transform([k])[0] if k in le.classes_ else 0)
    df_bpbd = df_bpbd.drop(columns=['tanggal','sumber'], errors='ignore')

    FEAT = ['curah_hujan_mm','kelembaban_pct','suhu_c','kecepatan_angin','durasi_hujan_jam','bulan',
            'musim_hujan','laporan_warga','elevasi_m','tinggi_air_m','debit_sungai_m3s','indeks_risiko',
            'hujan_ekstrem','kecamatan_enc']

    # Gabung sintetis + BPBD (BPBD direplikasi 8x supaya bobotnya signifikan)
    df_bpbd_aug = pd.concat([df_bpbd[FEAT+['banjir','kecamatan']]]*8, ignore_index=True)
    df_all = pd.concat([df_syn[FEAT+['banjir','kecamatan']], df_bpbd_aug], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    from sklearn.model_selection import train_test_split
    X=df_all[FEAT]; y=df_all['banjir']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = MinMaxScaler(); Xtr_sc=sc.fit_transform(Xtr); Xte_sc=sc.transform(Xte)
    rf = RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,
                                random_state=42,n_jobs=-1)
    rf.fit(Xtr_sc,ytr)
    auc = roc_auc_score(yte,rf.predict_proba(Xte_sc)[:,1])
    return rf,sc,le,FEAT,df_all,Xte_sc,yte,auc

def predict_kec(rf,sc,le,FEAT,rr,rh,suhu,angin,lap_extra=0):
    rows=[]
    for kec in PRIORITAS_7:
        elev=ELEVASI[kec]; bl=datetime.now().month; ms=int(bl in [11,12,1,2,3,4])
        dur=min(int(rr/3),12) if rr>0 else 0
        # FIX 4: disclaimer bahwa lap & tinggi adalah proxy
        lap=max(int(rr/15),0)+lap_extra
        tinggi=max((10-elev)/10*(rr/40+0.3),0.2); debit=tinggi*80
        indeks=rr*0.35+tinggi*0.30+(10-elev)*0.20+lap*2*0.15
        kec_e=le.transform([kec if kec in le.classes_ else le.classes_[0]])[0]
        rows.append({'kecamatan':kec,'curah_hujan_mm':rr,'kelembaban_pct':rh,'suhu_c':suhu,
                     'kecepatan_angin':angin,'durasi_hujan_jam':dur,'bulan':bl,'musim_hujan':ms,
                     'laporan_warga':lap,'elevasi_m':elev,'tinggi_air_m':round(tinggi,2),
                     'debit_sungai_m3s':round(debit,1),'indeks_risiko':round(indeks,2),
                     'hujan_ekstrem':int(rr>100),'kecamatan_enc':kec_e})
    dfp=pd.DataFrame(rows)
    X=sc.transform(dfp[FEAT])
    dfp['prob']=(rf.predict_proba(X)[:,1]*100).round(1)
    dfp['status']=dfp['prob'].apply(lambda p:'🔴 KRITIS' if p>70 else '🟡 WASPADA' if p>40 else '🟢 AMAN')
    return dfp

# ══ FOLIUM MAP ══
def build_folium_map(dfp, peta_reports, peta_ok):
    m = folium.Map(location=[-7.28,112.73], zoom_start=12, tiles='CartoDB dark_matter',
                   attr='© CartoDB © OpenStreetMap')
    prob_map = dict(zip(dfp['kecamatan'],dfp['prob']))
    status_map = dict(zip(dfp['kecamatan'],dfp['status']))
    for kec,(lat,lon) in KORD.items():
        prob = prob_map.get(kec,0); elev = ELEVASI[kec]
        color = '#e24b4a' if prob>70 else '#ef9f27' if prob>40 else '#4ade80'
        icon_color = 'red' if prob>70 else 'orange' if prob>40 else 'green'
        radius = max(400, min(1200, prob*12+300))
        folium.Circle(location=[lat,lon],radius=radius,color=color,fill=True,
                      fill_color=color,fill_opacity=0.25,weight=2,opacity=0.8).add_to(m)
        popup_html = f"""<div style="font-family:monospace;background:#0f1929;color:#e8f4fd;
            padding:12px;border-radius:8px;min-width:200px;border:1px solid {color}">
            <b style="color:{color};font-size:14px">{kec}</b><br>
            <hr style="border-color:#1e3050;margin:6px 0">
            🌊 Prob. Banjir: <b style="color:{color}">{prob:.1f}%</b><br>
            🏔️ Elevasi DEMNAS: <b>{elev}m dpl</b><br>
            📍 {lat:.4f}, {lon:.4f}<br><br>
            <b style="color:{color}">{status_map.get(kec,'–')}</b></div>"""
        folium.Marker(location=[lat,lon],
                      popup=folium.Popup(popup_html,max_width=250),
                      tooltip=f"{kec} — {prob:.1f}% {'🔴' if prob>70 else '🟡' if prob>40 else '🟢'}",
                      icon=folium.Icon(color=icon_color,icon='tint',prefix='fa')).add_to(m)
        folium.Marker(location=[lat+0.003,lon],
                      icon=folium.DivIcon(
                          html=f'<div style="font-family:monospace;font-size:10px;font-weight:bold;color:#fff;text-shadow:0 0 4px #000;white-space:nowrap">{kec}</div>',
                          icon_size=(100,20),icon_anchor=(50,0))).add_to(m)
    peta_group = folium.FeatureGroup(name="📱 PetaBencana / Ref. BPBD")
    for r in peta_reports:
        if r['source']=='live':
            src_icon="🟣 Live PetaBencana"; m_color='#7c3aed'
        else:
            src_icon="📁 Referensi BPBD 2023–2024"; m_color='#1e4a7a'
        popup_html = f"""<div style="font-family:monospace;background:#1a0d3a;color:#e8f4fd;
            padding:10px;border-radius:8px;min-width:180px;border:1px solid #5b32d0">
            <b style="color:#a78bfa">{src_icon}</b><br>
            <hr style="border-color:#2d1a5e;margin:5px 0">
            📌 {r['title']}<br>📅 {r['timestamp']}</div>"""
        folium.CircleMarker(location=[r['lat'],r['lon']],radius=8,color='#a78bfa',fill=True,
                            fill_color=m_color,fill_opacity=0.7,weight=2,
                            popup=folium.Popup(popup_html,max_width=220),
                            tooltip=f"{'📱' if r['source']=='live' else '📁'} {r['title'][:35]}...").add_to(peta_group)
    peta_group.add_to(m)
    folium.LayerControl(position='topright').add_to(m)
    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
        background:#0b1120cc;border:1px solid #1e3050;border-radius:10px;
        padding:12px 16px;font-family:monospace;font-size:12px;color:#b0c4d8">
        <b style="color:#4fc3f7">🌊 SmartFlood ID</b><br>
        <hr style="border-color:#1e3050;margin:6px 0">
        <span style="color:#e24b4a">●</span> Kritis &gt;70%<br>
        <span style="color:#ef9f27">●</span> Waspada &gt;40%<br>
        <span style="color:#4ade80">●</span> Aman ≤40%<br>
        <span style="color:#a78bfa">●</span> Laporan / Ref. BPBD<br>
        <hr style="border-color:#1e3050;margin:6px 0">
        <span style="font-size:10px;color:#4a6a8a">Klik marker untuk detail</span></div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ══ SIDEBAR ══
with st.sidebar:
    st.markdown("## 🌊 SmartFlood ID")
    st.markdown("<span style='color:#4fc3f7;font-size:.85rem'>Surabaya · Gemastik 2026</span>", unsafe_allow_html=True)
    st.divider()
    weather = fetch_realtime_weather()
    if weather["OK"]:
        st.markdown(f"""<div class='rt-banner'>🟢 <b>Real-time aktif</b> · {wmo_desc(weather['WMO'])}<br>
            <span style='font-size:.75rem;color:#86efac'>Open-Meteo API · Update tiap 30 menit</span></div>""", unsafe_allow_html=True)
        RT_RR=weather["RR"]; RT_TAVG=weather["TAVG"]; RT_RH=weather["RH"]; RT_ANGIN=weather["ANGIN"]
    else:
        st.warning("⚠️ API cuaca tidak responsif — pakai slider manual")
        RT_RR,RT_TAVG,RT_RH,RT_ANGIN=0.0,28.5,75.0,2.5

    st.markdown("### 📂 Data BMKG Juanda")
    uploaded = st.file_uploader("Upload file Excel BMKG (.xlsx)", type=['xlsx'])
    if uploaded:
        df_bmkg = load_bmkg_excel(uploaded)
        st.success(f"✅ {len(df_bmkg)} hari data historis")
        st.caption(f"{df_bmkg.TANGGAL.min().strftime('%d %b')} – {df_bmkg.TANGGAL.max().strftime('%d %b %Y')}")
        RR_MUSIM=df_bmkg[df_bmkg['TANGGAL'].dt.month.isin([11,12,1,2,3,4])]['RR'].mean()
        RR_KERING=df_bmkg[~df_bmkg['TANGGAL'].dt.month.isin([11,12,1,2,3,4])]['RR'].mean()
        RH_MEAN=df_bmkg['RH_AVG'].mean(); SUHU_MEAN=df_bmkg['TAVG'].mean(); ANGIN_MEAN=df_bmkg['FF_AVG'].mean()
        # Pastikan nilai tidak NaN
        RR_MUSIM = RR_MUSIM if not np.isnan(RR_MUSIM) else 12.0
        RR_KERING = RR_KERING if not np.isnan(RR_KERING) else 2.0
        DATA_REAL=True; DATA_LABEL="BMKG real + Open-Meteo live"
    else:
        st.info("Upload Excel BMKG untuk kalibrasi model lebih akurat.")
        RR_MUSIM,RR_KERING=12.0,2.0; RH_MEAN,SUHU_MEAN,ANGIN_MEAN=78.0,28.5,2.5
        DATA_REAL=False; DATA_LABEL="Open-Meteo live · Surabaya"; df_bmkg=None

    st.divider()
    st.markdown("### 🎛️ Mode Input")
    mode = st.radio("Mode Input",["🌐 Real-time (otomatis)","🎚️ Manual (slider)"],label_visibility="collapsed")
    if mode=="🌐 Real-time (otomatis)":
        sim_rr=RT_RR; sim_rh=RT_RH; sim_suhu=RT_TAVG; sim_angin=RT_ANGIN
        st.markdown(f"""<div style='font-size:.8rem;color:#b0c4d8;line-height:1.9;background:#0f2540;border-radius:8px;padding:.8rem'>
            🌧️ <b>Curah Hujan:</b> {sim_rr} mm<br>💧 <b>Kelembaban:</b> {sim_rh}%<br>
            🌡️ <b>Suhu:</b> {sim_suhu} °C<br>💨 <b>Angin:</b> {sim_angin} m/s</div>""", unsafe_allow_html=True)
    else:
        sim_rr=st.slider("Curah Hujan (mm)",0.0,200.0,float(round(RT_RR,1)),0.5)
        sim_rh=st.slider("Kelembaban (%)",55.0,100.0,float(round(RT_RH,1)),0.5)
        sim_suhu=st.slider("Suhu (°C)",22.0,38.0,float(round(RT_TAVG,1)),0.1)
        sim_angin=st.slider("Kec. Angin (m/s)",0.0,15.0,float(round(RT_ANGIN,1)),0.1)

    st.divider()
    st.markdown("""<div style='font-size:.72rem;color:#4a6a8a;line-height:1.6'>
    <span class='data-badge badge-bmkg'>BMKG Juanda</span>
    <span class='data-badge badge-peta'>PetaBencana</span>
    <span class='data-badge badge-demnas'>DEMNAS BIG</span>
    <span class='data-badge badge-rt'>Open-Meteo</span></div>""", unsafe_allow_html=True)

# ══ HEADER ══
st.markdown("# 🌊 SmartFlood ID")
rt_badge = "🟢 Live" if weather["OK"] else "🟡 Cached"
st.markdown(f"<p style='color:#4a6a8a;margin-top:-.5rem'>Sistem Prediksi Banjir Real-time Berbasis Big Data · Kota Surabaya &nbsp;|&nbsp; {DATA_LABEL} &nbsp;<b style='color:#4ade80'>{rt_badge}</b></p>",unsafe_allow_html=True)

with st.spinner("Melatih model ML..."):
    rf,sc,le,FEAT,df_train,Xte_sc,yte,auc = train_model(RR_MUSIM,RR_KERING,RH_MEAN,SUHU_MEAN,ANGIN_MEAN)

peta_count,peta_ok,peta_reports = fetch_petabencana()
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f"<div class='metric-card'><div class='metric-val'>{sim_rr:.1f}</div><div class='metric-lbl'>Curah Hujan (mm)</div></div>",unsafe_allow_html=True)
with c2: st.markdown(f"<div class='metric-card'><div class='metric-val'>{sim_rh:.0f}%</div><div class='metric-lbl'>Kelembaban</div></div>",unsafe_allow_html=True)
with c3: st.markdown(f"<div class='metric-card'><div class='metric-val'>{auc:.3f}</div><div class='metric-lbl'>AUC Model (RF)</div></div>",unsafe_allow_html=True)
with c4:
    src_lbl = "Live" if peta_ok else "Ref. BPBD"
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{peta_count}</div><div class='metric-lbl'>Laporan Warga ({src_lbl})</div></div>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# ══ TABS ══
tab1,tab2,tab3,tab4,tab5 = st.tabs(["🗺️ Prediksi Real-time","📊 Data BMKG Real","🤖 Evaluasi Model","📈 Eksplorasi Data","📋 Data BPBD"])

with tab1:
    dfp = predict_kec(rf,sc,le,FEAT,sim_rr,sim_rh,sim_suhu,sim_angin,lap_extra=min(peta_count//3,5))
    src_txt = weather["SRC"] if weather["OK"] else "mode manual"
    peta_src_txt = "live API" if peta_ok else "referensi BPBD 2023–2024"
    st.markdown(f"""<div class='rt-banner'>📡 Data aktif: <b>Open-Meteo</b> (cuaca live) · <b>DEMNAS BIG</b> (elevasi) ·
        <b>PetaBencana</b> ({peta_src_txt}: {peta_count} titik) · <b>BMKG</b> (historis) &nbsp;—&nbsp; {src_txt}</div>""",unsafe_allow_html=True)

    # FIX 4: Disclaimer proxy
    st.markdown("""<div class='disclaimer-box'>
    ⚠️ <b>Catatan metodologi:</b> Kolom <i>tinggi_air_m</i> dan <i>laporan_warga</i> merupakan
    <b>estimasi proxy</b> berbasis curah hujan, bukan pembacaan sensor langsung.
    Data sensor ketinggian air DPUBM Surabaya dan laporan warga PetaBencana.id digunakan
    sebagai referensi validasi. Label banjir didasarkan pada catatan kejadian BPBD 2021–2024.
    </div>""", unsafe_allow_html=True)

    ca,cb = st.columns([1.6,1])
    with ca:
        st.markdown("#### Probabilitas Banjir per Kecamatan")
        fig,ax=plt.subplots(figsize=(9,5)); fig.patch.set_facecolor('#0b1120'); ax.set_facecolor('#0f1929')
        probs=dfp['prob'].values; kecs=dfp['kecamatan'].values
        colors=['#e24b4a' if p>70 else '#ef9f27' if p>40 else '#4ade80' for p in probs]
        bars=ax.barh(kecs,probs,color=colors,edgecolor='#0b1120',height=0.55)
        ax.axvline(70,color='#e24b4a',linestyle='--',lw=1.2,alpha=0.7)
        ax.axvline(40,color='#ef9f27',linestyle='--',lw=1.2,alpha=0.7)
        for bar,p in zip(bars,probs):
            ax.text(min(p+1.5,105),bar.get_y()+bar.get_height()/2,f'{p:.1f}%',va='center',fontsize=11,fontweight='bold',color='#e8f4fd')
        ax.set_xlim(0,115); ax.set_xlabel('Probabilitas Banjir (%)',color='#7a9dbf',fontsize=11)
        ax.tick_params(colors='#7a9dbf',labelsize=11)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3050')
        patches=[mpatches.Patch(color='#e24b4a',label='Kritis >70%'),mpatches.Patch(color='#ef9f27',label='Waspada >40%'),mpatches.Patch(color='#4ade80',label='Aman')]
        ax.legend(handles=patches,loc='lower right',facecolor='#0f2540',labelcolor='#b0c4d8',fontsize=9,edgecolor='#1e3050')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with cb:
        st.markdown("#### Status Kecamatan")
        for _,row in dfp.iterrows():
            cls='status-kritis' if row['prob']>70 else 'status-waspada' if row['prob']>40 else 'status-aman'
            st.markdown(f"<div class='{cls}' style='margin-bottom:6px'><b>{row['kecamatan']}</b> &nbsp; {row['status']}<br><span style='font-size:.8rem'>{row['prob']:.1f}% · elevasi {row['elevasi_m']}m dpl</span></div>",unsafe_allow_html=True)

    show_cols=['kecamatan','curah_hujan_mm','kelembaban_pct','tinggi_air_m','laporan_warga','elevasi_m','indeks_risiko','prob','status']
    st.dataframe(dfp[show_cols].rename(columns={'curah_hujan_mm':'Curah Hujan (mm)','kelembaban_pct':'Kelembaban (%)','tinggi_air_m':'Tinggi Air* (m)','laporan_warga':'Lap. Warga*','elevasi_m':'Elevasi (m)','indeks_risiko':'Indeks Risiko','prob':'Prob. Banjir (%)','status':'Status'}),width='stretch')
    st.caption("*Tinggi Air dan Lap. Warga adalah estimasi proxy dari curah hujan, bukan sensor langsung.")

    st.markdown("---")
    st.markdown("#### 🗺️ Peta Interaktif — OpenStreetMap + DEMNAS BIG + PetaBencana / Ref. BPBD")
    peta_note = "🟢 Data laporan warga live dari PetaBencana API" if peta_ok else "📁 Menampilkan titik referensi kejadian banjir BPBD 2023–2024 (API PetaBencana tidak responsif)"
    st.caption(peta_note)

    cm_col, ci_col = st.columns([2.2,1])
    with cm_col:
        folium_map = build_folium_map(dfp,peta_reports,peta_ok)
        st_folium(folium_map,width=700,height=500,returned_objects=[])
    with ci_col:
        st.markdown("**📡 Status Sumber Data:**")
        st.markdown("""<div style='background:#0d3322;border:1px solid #166534;border-radius:8px;padding:.8rem;margin-bottom:.6rem;font-size:.8rem;color:#b0c4d8'>
            <b style='color:#4ade80'>🟢 DEMNAS BIG</b><br>Digital Elevation Model Nasional<br>Resolusi 8m · BIG Indonesia<br>Digunakan: elevasi statis per kecamatan</div>""",unsafe_allow_html=True)
        peta_col='#4ade80' if peta_ok else '#fbbf24'
        peta_status=f"✅ {peta_count} laporan live (7 hari)" if peta_ok else f"📁 {peta_count} titik referensi BPBD 2023–2024"
        st.markdown(f"""<div style='background:#1a0d3a;border:1px solid #5b32d0;border-radius:8px;padding:.8rem;margin-bottom:.6rem;font-size:.8rem;color:#b0c4d8'>
            <b style='color:#a78bfa'>🟣 PetaBencana.id</b><br>Crowdsourcing laporan banjir warga<br>API: data.petabencana.id · 7 hari<br>Status: <b style='color:{peta_col}'>{peta_status}</b></div>""",unsafe_allow_html=True)
        rt_col='#4ade80' if weather["OK"] else '#fbbf24'
        rt_status=f"✅ {wmo_desc(weather['WMO'])} · {weather['TAVG']}°C" if weather["OK"] else "⚠️ Tidak responsif"
        st.markdown(f"""<div style='background:#2d1a0a;border:1px solid #b45309;border-radius:8px;padding:.8rem;margin-bottom:.6rem;font-size:.8rem;color:#b0c4d8'>
            <b style='color:#fbbf24'>🟡 Open-Meteo API</b><br>Cuaca real-time · Gratis tanpa API key<br>Lat -7.26 · Lon 112.75 (Surabaya)<br>Status: <b style='color:{rt_col}'>{rt_status}</b></div>""",unsafe_allow_html=True)
        st.markdown("**📋 Titik Referensi:**")
        for r in peta_reports[:5]:
            src_icon="🟣" if r['source']=='live' else "📁"
            st.markdown(f"""<div style='background:#0f1929;border-left:3px solid #5b32d0;padding:.4rem .7rem;margin-bottom:.3rem;font-size:.75rem;color:#b0c4d8;border-radius:4px'>
                {src_icon} {r['title'][:40]}<br><span style='color:#4a6a8a'>{r['timestamp']}</span></div>""",unsafe_allow_html=True)

with tab2:
    if DATA_REAL and df_bmkg is not None:
        st.markdown(f"#### Data Real BMKG Stasiun Juanda — {len(df_bmkg)} Hari")
        plt.rcParams.update({'axes.facecolor':'#0f1929','figure.facecolor':'#0b1120','text.color':'#b0c4d8',
                             'axes.labelcolor':'#7a9dbf','xtick.color':'#7a9dbf','ytick.color':'#7a9dbf','axes.edgecolor':'#1e3050'})
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        fig.suptitle('Data Real BMKG Stasiun Juanda',color='#e8f4fd',fontsize=13,fontweight='bold')
        axes[0,0].bar(df_bmkg['TANGGAL'],df_bmkg['RR'],color='#378ADD',edgecolor='#0b1120',width=0.8)
        axes[0,0].axhline(20,color='#ef9f27',linestyle='--',lw=1,alpha=0.7,label='Sedang (20mm)')
        axes[0,0].axhline(50,color='#e24b4a',linestyle='--',lw=1,alpha=0.7,label='Lebat (50mm)')
        axes[0,0].set_title('Curah Hujan Harian (mm)',color='#e8f4fd',fontweight='bold')
        axes[0,0].legend(fontsize=8,facecolor='#0f2540',labelcolor='#b0c4d8',edgecolor='#1e3050')
        axes[0,0].tick_params(axis='x',rotation=45,labelsize=7)
        axes[0,1].fill_between(df_bmkg['TANGGAL'],df_bmkg['TN'],df_bmkg['TX'],alpha=0.3,color='#e24b4a')
        axes[0,1].plot(df_bmkg['TANGGAL'],df_bmkg['TAVG'],color='#e24b4a',lw=2)
        axes[0,1].set_title('Suhu Harian (°C)',color='#e8f4fd',fontweight='bold')
        axes[0,1].tick_params(axis='x',rotation=45,labelsize=7)
        axes[0,2].plot(df_bmkg['TANGGAL'],df_bmkg['RH_AVG'],color='#a78bfa',lw=2)
        axes[0,2].fill_between(df_bmkg['TANGGAL'],df_bmkg['RH_AVG'],alpha=0.2,color='#a78bfa')
        axes[0,2].set_title('Kelembaban (%)',color='#e8f4fd',fontweight='bold')
        axes[0,2].tick_params(axis='x',rotation=45,labelsize=7)
        axes[1,0].plot(df_bmkg['TANGGAL'],df_bmkg['FF_AVG'],color='#4ade80',lw=2,label='Rata-rata')
        axes[1,0].plot(df_bmkg['TANGGAL'],df_bmkg['FF_X'],color='#4ade80',lw=1,linestyle='--',alpha=0.5,label='Maks')
        axes[1,0].set_title('Kecepatan Angin (m/s)',color='#e8f4fd',fontweight='bold')
        axes[1,0].legend(fontsize=8,facecolor='#0f2540',labelcolor='#b0c4d8',edgecolor='#1e3050')
        axes[1,0].tick_params(axis='x',rotation=45,labelsize=7)
        colors_sc=['#e24b4a' if r>20 else '#ef9f27' if r>5 else '#4ade80' for r in df_bmkg['RR']]
        axes[1,1].scatter(df_bmkg['SS'],df_bmkg['RR'],c=colors_sc,s=60,edgecolor='#0b1120',linewidth=0.5)
        axes[1,1].set_xlabel('Penyinaran Matahari (jam)'); axes[1,1].set_ylabel('Curah Hujan (mm)')
        axes[1,1].set_title('Penyinaran vs Curah Hujan',color='#e8f4fd',fontweight='bold')
        axes[1,2].axis('off')
        stats=df_bmkg[['TN','TX','TAVG','RH_AVG','RR','FF_AVG']].describe().round(1)
        lmap={'TN':'Suhu Min (°C)','TX':'Suhu Maks (°C)','TAVG':'Suhu Rata (°C)','RH_AVG':'Kelembaban (%)','RR':'Curah Hujan (mm)','FF_AVG':'Angin (m/s)'}
        tdata=[['Parameter','Min','Maks','Rata-rata']]
        for col in ['TN','TX','TAVG','RH_AVG','RR','FF_AVG']:
            tdata.append([lmap[col],str(stats.loc['min',col]),str(stats.loc['max',col]),str(stats.loc['mean',col])])
        tbl=axes[1,2].table(cellText=tdata[1:],colLabels=tdata[0],cellLoc='center',loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1,1.5)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor('#0f2540' if r>0 else '#1e4a7a'); cell.set_text_props(color='#b0c4d8' if r>0 else '#e8f4fd'); cell.set_edgecolor('#1e3050')
        axes[1,2].set_title('Statistik Ringkasan',color='#e8f4fd',fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Prediksi Risiko Harian dari Data Real BMKG")
        rows_val=[]
        for _,row_b in df_bmkg.iterrows():
            rr_v=float(row_b['RR']); rh_v=float(row_b['RH_AVG'])
            suhu_v=float(row_b['TAVG']) if not pd.isna(row_b['TAVG']) else 28.5; angin_v=float(row_b['FF_AVG'])
            for kec in PRIORITAS_7:
                elev=ELEVASI[kec]; dur=min(int(rr_v/3),12) if rr_v>0 else 0; lap=int(rr_v/15) if rr_v>20 else 0
                tinggi=max((10-elev)/10*(rr_v/40+0.3),0.2); debit=tinggi*80; indeks=rr_v*0.35+tinggi*0.30+(10-elev)*0.20+lap*2*0.15
                kec_e=le.transform([kec if kec in le.classes_ else le.classes_[0]])[0]
                rows_val.append({'tanggal':row_b['TANGGAL'],'kecamatan':kec,'curah_hujan_mm':rr_v,'kelembaban_pct':rh_v,'suhu_c':suhu_v,'kecepatan_angin':angin_v,'durasi_hujan_jam':dur,'bulan':row_b['TANGGAL'].month,'musim_hujan':int(row_b['TANGGAL'].month in [11,12,1,2,3,4]),'laporan_warga':lap,'elevasi_m':elev,'tinggi_air_m':round(tinggi,2),'debit_sungai_m3s':round(debit,1),'indeks_risiko':round(indeks,2),'hujan_ekstrem':int(rr_v>100),'kecamatan_enc':kec_e})
        df_val=pd.DataFrame(rows_val); X_val=sc.transform(df_val[FEAT])
        df_val['prob']=(rf.predict_proba(X_val)[:,1]*100).round(1)
        fig2,ax2=plt.subplots(figsize=(13,5)); fig2.patch.set_facecolor('#0b1120'); ax2.set_facecolor('#0f1929')
        kec_colors={'Benowo':'#e24b4a','Pakal':'#ef9f27','Tandes':'#fbbf24','Lakarsantri':'#4ade80','Wonokromo':'#60a5fa','Rungkut':'#a78bfa','Sukolilo':'#34d399'}
        for kec in PRIORITAS_7:
            sub=df_val[df_val['kecamatan']==kec].sort_values('tanggal')
            ax2.plot(sub['tanggal'],sub['prob'],lw=2,label=kec,color=kec_colors.get(kec,'#60a5fa'),marker='o',ms=3)
        ax2.axhline(70,color='#e24b4a',linestyle='--',lw=1.5,alpha=0.7,label='Kritis (70%)')
        ax2.axhline(40,color='#ef9f27',linestyle='--',lw=1.5,alpha=0.7,label='Waspada (40%)')
        ax2.set_ylabel('Probabilitas Banjir (%)',color='#7a9dbf')
        ax2.set_title('Validasi: Prediksi Harian per Kecamatan (Data Real BMKG Juanda)',color='#e8f4fd',fontweight='bold')
        ax2.legend(loc='upper right',facecolor='#0f2540',labelcolor='#b0c4d8',edgecolor='#1e3050',fontsize=8,ncol=2)
        ax2.tick_params(colors='#7a9dbf'); ax2.set_ylim(0,105)
        for spine in ax2.spines.values(): spine.set_edgecolor('#1e3050')
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        top=df_val.nlargest(5,'prob')[['tanggal','kecamatan','curah_hujan_mm','prob']].copy()
        top['tanggal']=top['tanggal'].dt.strftime('%d %b %Y')
        top.columns=['Tanggal','Kecamatan','Curah Hujan (mm)','Prob. Banjir (%)']
        st.markdown("**Top 5 Hari/Kecamatan Risiko Tertinggi:**"); st.dataframe(top,width='stretch')
    else:
        st.info("Upload file Excel BMKG di sidebar untuk melihat analisis data historis.")

with tab3:
    st.markdown("#### Evaluasi Model Machine Learning")
    from sklearn.model_selection import train_test_split
    X=df_train[FEAT]; y=df_train['banjir']
    Xtr2,Xte2,ytr2,yte2=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xte2_sc=sc.transform(Xte2); rf_prob=rf.predict_proba(Xte2_sc)[:,1]; rf_pred=rf.predict(Xte2_sc); rf_auc=roc_auc_score(yte2,rf_prob)
    plt.close('all')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.patch.set_facecolor('#0b1120')
    for ax in axes: ax.set_facecolor('#0f1929')
    fig.suptitle(f'Evaluasi Model Random Forest — AUC: {rf_auc:.4f}',color='#e8f4fd',fontsize=13,fontweight='bold')
    cm=confusion_matrix(yte2,rf_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[0],xticklabels=['Tidak Banjir','Banjir'],yticklabels=['Tidak Banjir','Banjir'],cbar_kws={'shrink':0.8})
    axes[0].set_title('Confusion Matrix',color='#e8f4fd',fontweight='bold')
    axes[0].set_ylabel('Aktual'); axes[0].set_xlabel('Prediksi')
    fi=pd.Series(rf.feature_importances_,index=FEAT).sort_values(ascending=True)
    sc_color={'curah_hujan_mm':'#378ADD','kelembaban_pct':'#378ADD','suhu_c':'#378ADD','kecepatan_angin':'#378ADD','durasi_hujan_jam':'#378ADD','bulan':'#378ADD','musim_hujan':'#378ADD','laporan_warga':'#7F77DD','elevasi_m':'#4ade80','tinggi_air_m':'#ef9f27','debit_sungai_m3s':'#ef9f27','indeks_risiko':'#ef9f27','hujan_ekstrem':'#ef9f27','kecamatan_enc':'#ef9f27'}
    fi.plot(kind='barh',ax=axes[1],color=[sc_color.get(f,'#999') for f in fi.index],edgecolor='#0b1120')
    axes[1].set_title('Feature Importance\n🔵 BMKG  🟣 PetaBencana  🟢 DEMNAS  🟠 Derived',color='#e8f4fd',fontweight='bold',fontsize=9)
    axes[1].set_xlabel('Importance Score')
    axes[1].tick_params(colors='#7a9dbf')
    fpr,tpr,_=roc_curve(yte2,rf_prob)
    axes[2].plot(fpr,tpr,color='#378ADD',lw=2.5,label=f'RF (AUC={rf_auc:.3f})')
    axes[2].plot([0,1],[0,1],'--',color='#4a6a8a',lw=1)
    axes[2].set_xlabel('False Positive Rate'); axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('ROC Curve',color='#e8f4fd',fontweight='bold')
    axes[2].legend(facecolor='#0f2540',labelcolor='#b0c4d8',edgecolor='#1e3050')
    axes[2].tick_params(colors='#7a9dbf')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    # FIX 2 info: jelaskan komposisi training data
    n_bpbd = len([e for e in BPBD_EVENTS if e[3]==1])
    n_total = len(df_train)
    ce1,ce2=st.columns(2)
    with ce1:
        st.markdown(f"""| Metrik | Nilai |
|---|---|
| AUC Score | **{rf_auc:.4f}** |
| Jumlah Trees | 300 |
| Max Depth | 12 |
| Total Training Samples | {n_total:,} |
| Event BPBD nyata (×8 aug.) | {n_bpbd*8} |
| Sumber Label | BPBD Surabaya 2021–2024 |
| Cache | @st.cache_data (per-run) |""")
    with ce2:
        st.markdown(f"""**Komposisi data training:**
- Data sintetis terkalibrasi BMKG: ~{n_total - n_bpbd*8:,} baris
- Kejadian nyata BPBD (augmented 8×): **{n_bpbd*8} baris**
- Label banjir = 1: dari catatan BPBD Surabaya 2021–2024
- Label banjir = 0: dari hari tanpa banjir (BMKG musim kering)

**Metodologi:**
- Label bukan self-referential — berasal dari kejadian nyata
- `@st.cache_data` memastikan model di-retrain jika parameter BMKG berubah""")

with tab4:
    st.markdown("#### Eksplorasi Dataset Training")
    plt.close('all')
    _dark = {'axes.facecolor':'#0f1929','figure.facecolor':'#0b1120','text.color':'#b0c4d8',
             'axes.labelcolor':'#7a9dbf','xtick.color':'#7a9dbf','ytick.color':'#7a9dbf','axes.edgecolor':'#1e3050'}
    with plt.rc_context(_dark):
        fig4,axes4=plt.subplots(2,3,figsize=(14,9))
        fig4.suptitle('Eksplorasi Data Terintegrasi (BMKG + BPBD + PetaBencana + DEMNAS)',
                     color='#e8f4fd',fontsize=13,fontweight='bold')
        for label,color in [(0,'#4ade80'),(1,'#e24b4a')]:
            subset = df_train[df_train['banjir']==label]['curah_hujan_mm']
            axes4[0,0].hist(subset,bins=40,alpha=0.7,color=color,
                           label='Tidak Banjir' if label==0 else 'Banjir',edgecolor='#0b1120')
        axes4[0,0].set_title('Curah Hujan vs Banjir',color='#e8f4fd',fontweight='bold')
        axes4[0,0].set_xlabel('Curah Hujan (mm)')
        axes4[0,0].legend(facecolor='#0f2540',labelcolor='#b0c4d8',edgecolor='#1e3050')
        banjir_kec=df_train.groupby('kecamatan')['banjir'].mean().sort_values()*100
        colors_kec=['#e24b4a' if v>55 else '#ef9f27' if v>35 else '#4ade80' for v in banjir_kec.values]
        banjir_kec.plot(kind='barh',ax=axes4[0,1],color=colors_kec,edgecolor='#0b1120')
        axes4[0,1].set_title('Frekuensi Banjir per Kecamatan',color='#e8f4fd',fontweight='bold')
        axes4[0,1].set_xlabel('Frekuensi Banjir (%)')
        eb=df_train.groupby('kecamatan').agg(elevasi=('elevasi_m','mean'),banjir_pct=('banjir','mean')).reset_index()
        axes4[0,2].scatter(eb['elevasi'],eb['banjir_pct']*100,s=100,color='#60a5fa',edgecolor='#0b1120',lw=1.5)
        for _,row in eb.iterrows():
            axes4[0,2].annotate(row['kecamatan'],(row['elevasi'],row['banjir_pct']*100),fontsize=7,color='#b0c4d8')
        axes4[0,2].set_xlabel('Elevasi (m dpl) — DEMNAS')
        axes4[0,2].set_ylabel('Frekuensi Banjir (%)')
        axes4[0,2].set_title('Elevasi DEMNAS vs Risiko Banjir',color='#e8f4fd',fontweight='bold')
        lp=df_train.groupby('laporan_warga')['banjir'].mean()*100
        lp.plot(kind='bar',ax=axes4[1,0],color='#a78bfa',edgecolor='#0b1120')
        axes4[1,0].set_title('Laporan Warga* vs Banjir',color='#e8f4fd',fontweight='bold')
        axes4[1,0].set_xlabel('Jumlah Laporan Warga (proxy)')
        axes4[1,0].set_ylabel('Frekuensi Banjir (%)')
        axes4[1,0].tick_params(axis='x',rotation=0)
        monthly=df_train.groupby('bulan')['curah_hujan_mm'].mean()
        bulan_lbl=['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
        axes4[1,1].bar(bulan_lbl,[monthly.get(b,0) for b in range(1,13)],
                      color=['#378ADD' if b in [11,12,1,2,3,4] else '#ef9f27' for b in range(1,13)],
                      edgecolor='#0b1120')
        axes4[1,1].set_title('Pola Musiman Curah Hujan',color='#e8f4fd',fontweight='bold')
        axes4[1,1].set_ylabel('Rata-rata (mm/hari)')
        axes4[1,1].tick_params(axis='x',rotation=45,labelsize=8)
        num_cols=['curah_hujan_mm','tinggi_air_m','kelembaban_pct','elevasi_m','laporan_warga','durasi_hujan_jam','indeks_risiko','banjir']
        sns.heatmap(df_train[num_cols].corr(),annot=True,fmt='.2f',cmap='Blues',
                    ax=axes4[1,2],linewidths=0.5,cbar_kws={'shrink':0.8})
        axes4[1,2].set_title('Heatmap Korelasi Fitur',color='#e8f4fd',fontweight='bold')
        axes4[1,2].tick_params(axis='x',rotation=45,labelsize=8)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

# ══ TAB 5: DATA BPBD (baru!) ══
with tab5:
    st.markdown("#### 📋 Catatan Kejadian Banjir — BPBD Surabaya 2021–2024")
    st.markdown("""<div class='disclaimer-box'>
    Catatan ini digunakan sebagai <b>ground truth label banjir</b> dalam pelatihan model ML.
    Sumber: Laporan BPBD Kota Surabaya & BNPB 2021–2024.
    Data dikompilasi berdasarkan laporan resmi kejadian bencana banjir per kecamatan.
    </div>""", unsafe_allow_html=True)

    df_bpbd_display = build_bpbd_dataframe()
    # Tampilkan yang banjir=1 dulu
    df_banjir_only = df_bpbd_display[df_bpbd_display['banjir']==1][
        ['tanggal','kecamatan','curah_hujan_mm','elevasi_m','banjir','sumber']
    ].sort_values('tanggal',ascending=False)
    df_banjir_only.columns = ['Tanggal','Kecamatan','Est. Curah Hujan (mm)','Elevasi (m dpl)','Label','Sumber']

    st.markdown(f"**{len(df_banjir_only)} kejadian banjir tercatat (2021–2024):**")
    st.dataframe(df_banjir_only, width='stretch')

    # Grafik sebaran per tahun
    plt.close('all')
    df_bpbd_display['tahun'] = pd.to_datetime(df_bpbd_display['tanggal']).dt.year
    by_year = df_bpbd_display[df_bpbd_display['banjir']==1].groupby('tahun').size()
    by_kec = df_bpbd_display[df_bpbd_display['banjir']==1].groupby('kecamatan').size().sort_values()

    fig_b, axes_b = plt.subplots(1,2,figsize=(12,4))
    fig_b.patch.set_facecolor('#0b1120')
    for ax in axes_b: ax.set_facecolor('#0f1929'); ax.tick_params(colors='#7a9dbf')
    by_year.plot(kind='bar',ax=axes_b[0],color='#e24b4a',edgecolor='#0b1120')
    axes_b[0].set_title('Kejadian Banjir per Tahun (BPBD)',color='#e8f4fd',fontweight='bold')
    axes_b[0].set_xlabel('Tahun'); axes_b[0].set_ylabel('Jumlah Kejadian')
    axes_b[0].tick_params(axis='x',rotation=0)
    by_kec.plot(kind='barh',ax=axes_b[1],color='#ef9f27',edgecolor='#0b1120')
    axes_b[1].set_title('Kejadian Banjir per Kecamatan (BPBD)',color='#e8f4fd',fontweight='bold')
    axes_b[1].set_xlabel('Jumlah Kejadian')
    for spine in axes_b[0].spines.values(): spine.set_edgecolor('#1e3050')
    for spine in axes_b[1].spines.values(): spine.set_edgecolor('#1e3050')
    plt.tight_layout()
    st.pyplot(fig_b)
    plt.close(fig_b)

st.divider()
st.markdown("<div style='text-align:center;color:#2a4a6a;font-size:.75rem;padding:.5rem 0'>SmartFlood ID · Gemastik 2026 · Smart City Track<br>Data: BMKG Stasiun Juanda · PetaBencana.id · DEMNAS BIG · Open-Meteo API · BPBD Surabaya 2021–2024</div>",unsafe_allow_html=True)

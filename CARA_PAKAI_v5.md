# Cara Pakai Paket v5 — WAJIB DIBACA SEBELUM DEMO

## Kenapa ada file ini?
Notebook (`SmartFlood_v5_FIXED.ipynb`) memperbaiki bug: model sekarang dilatih dengan
4 fitur tambahan (`tma_sungai_m`, `pasang_surut_m`, `excess_drainase`, `hujan_ekstrem`)
yang sebelumnya dihitung tapi tidak pernah dipakai. `app.py` dan `spark/batch_prediction.py`
di paket ini SUDAH disesuaikan untuk fitur baru ini — tapi file model lama
(`smartflood_rf_model.pkl`) belum cocok lagi, makanya sengaja diganti nama jadi
`*_OLD_..._JANGAN_DIPAKAI.pkl` supaya tidak salah pakai.

## Langkah wajib sebelum jalankan dashboard
1. Buka `SmartFlood_v5_FIXED.ipynb` di Google Colab.
2. Pastikan 17 file BMKG (`data-bmkg.zip`) ada di folder Google Drive `big/` kamu (path yang dipakai notebook: `/content/drive/MyDrive/big/*.xlsx`).
3. Run All. Di bagian akhir notebook, dua file akan ter-download:
   - `smartflood_rf_model.pkl`
   - `smartflood_encoder.pkl`
4. Pindahkan kedua file itu ke folder root project ini (timpa/sandingkan, **nama harus persis** `smartflood_rf_model.pkl` dan `smartflood_encoder.pkl`).
5. Baru jalankan `streamlit run app.py`.

Kalau langkah 3–4 dilewati, dashboard akan menampilkan pesan error "Gagal memuat model"
(bukan crash diam-diam) — itu sengaja, supaya tidak ada yang demo pakai model yang salah.

## Yang berubah di notebook v5 (ringkasan)
- Bug fix: 4 fitur hidrologi yang diklaim v4 sekarang benar-benar dipakai model.
- SMOTE diuji eksplisit dan ditolak dengan bukti (lihat sel 6c).
- Analisis oracle ceiling (sel 6d) — menjelaskan secara jujur batas atas performa.
- Validasi eksternal curah hujan riil vs 5 kejadian banjir Surabaya yang diberitakan media (sel 6e).
- Kesimpulan (sel 9) ditulis ulang dengan angka hasil uji ulang langsung + jawaban lengkap soal sumber ground truth.

## Angka final yang sudah diverifikasi ulang (hasil run lokal, bukan klaim)
- AUC test-set: ~0.75 | AUC 5-fold CV: ~0.78
- F1 kelas Banjir: ~0.30–0.40 (tergantung fold/threshold)
- Model mencapai ~97% dari AUC oracle (batas teoretis skema label sintetis kami)
- 4 dari 5 kejadian banjir nyata (Surabaya, Nov 2024–Mei 2026) konsisten dengan curah hujan ekstrem di data BMKG kami

# Sistem MLOps untuk Prediksi Jangka Pendek Kualitas Udara (PM2.5) di Jakarta

Proyek ini merupakan Sistem MLOps (Machine Learning Operations) yang dirancang untuk memprediksi konsentrasi Particulate Matter 2.5 (PM2.5) di wilayah DKI Jakarta untuk jangka waktu pendek (+3 jam ke depan) menggunakan data historis selama enam jam terakhir (sliding window 6 jam)[cite: 1].

Sistem ini tidak hanya berfokus pada pembuatan model prediktif, tetapi juga pada pengelolaan siklus hidup model, termasuk pemantauan perubahan data (*feature drift*, *target drift*), deteksi penurunan performa, dan pelatihan ulang secara otomatis (*continuous training*)[cite: 1]. Model *baseline* yang digunakan pada tahap Proof-of-Concept (PoC) adalah *Linear Regression*[cite: 1].

## Tujuan Proyek
* Membangun rancangan sistem MLOps *end-to-end* untuk prediksi PM2.5 yang memiliki mekanisme *ingestion* data, *feature engineering* temporal, *training*, *inference*, *monitoring*, dan *continuous training*[cite: 1].
* Menghasilkan prediksi PM2.5 pada horizon +3 jam dengan menggunakan window historis 6 jam[cite: 1].
* Mendokumentasikan dan memonitor model menggunakan *Linear Regression* dengan metrik MAE dan RMSE[cite: 1].

## Struktur Direktori
Proyek ini mengikuti standar industri pengelolaan proyek data sains:

```text
├── configs/             # Konfigurasi pipeline dan parameter model
├── data/
│   ├── processed/       # Data yang telah melewati tahap preprocessing dan feature engineering
│   └── raw/             # Data mentah dari OpenWeatherMap Air Pollution API
├── docs/                # Dokumentasi tambahan proyek
├── models/              # Artefak model yang telah dilatih (champion & challenger)
├── notebooks/           # Jupyter notebooks untuk eksplorasi (EDA) dan eksperimen awal
├── src/                 # Source code utama untuk pipeline MLOps
│   ├── data/            # Script untuk akuisisi dan validasi data
│   ├── features/        # Script untuk rekayasa fitur (termasuk cyclical encoding)
│   └── models/          # Script untuk training, evaluasi, dan prediksi
├── tests/               # Unit testing untuk kode pipeline
├── .devcontainer/       # Konfigurasi lingkungan GitHub Codespaces
└── README.md            # Dokumentasi utama repositori
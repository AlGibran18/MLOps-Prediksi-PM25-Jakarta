# Sistem MLOps untuk Prediksi Jangka Pendek Kualitas Udara (PM2.5) di Jakarta

Proyek ini merupakan Sistem MLOps (Machine Learning Operations) yang dirancang untuk memprediksi konsentrasi Particulate Matter 2.5 (PM2.5) di wilayah DKI Jakarta untuk jangka waktu pendek (+3 jam ke depan) menggunakan data historis selama enam jam terakhir (sliding window 6 jam).

Sistem ini tidak hanya berfokus pada pembuatan model prediktif, tetapi juga pada pengelolaan siklus hidup model, termasuk pemantauan perubahan data (*feature drift*, *target drift*), deteksi penurunan performa, dan pelatihan ulang secara otomatis (*continuous training*). Model *baseline* yang digunakan pada tahap Proof-of-Concept (PoC) adalah *Linear Regression*.

## Tujuan Proyek
* Membangun rancangan sistem MLOps *end-to-end* untuk prediksi PM2.5 yang memiliki mekanisme *ingestion* data, *feature engineering* temporal, *training*, *inference*, *monitoring*, dan *continuous training*.
* Menghasilkan prediksi PM2.5 pada horizon +3 jam dengan menggunakan window historis 6 jam.
* Mendokumentasikan dan memonitor model menggunakan *Linear Regression* dengan metrik MAE dan RMSE.

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

```
## Cara Menjalankan Lingkungan Kerja (GitHub Codespaces)
Repositori ini dikonfigurasi menggunakan GitHub Codespaces untuk menjamin konsistensi *environment* tanpa masalah *dependency*.

1. Buka repositori ini di GitHub: https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta
2. Klik tombol hijau **Code**.
3. Pilih tab **Codespaces**.
4. Klik **Create codespace on main** (atau buka codespace yang sudah ada).
5. Tunggu beberapa saat hingga proses *build container* selesai (proses ini secara otomatis menginstal Python 3.10 dan pustaka MLOps seperti *scikit-learn, pandas, numpy, jupyter, dvc, mlflow,* dan *prometheus_client*).
6. Lingkungan kerja VS Code siap digunakan langsung dari browser Anda.

## Cara Menjalankan Pipeline Data
Sistem ini telah dilengkapi dengan skrip pengumpul data (Data Ingestion) secara otomatis terhadap sumber data dinamis, serta skrip automasi prapemrosesan. 

### 1. Penarikan Data Berkelanjutan (Data Ingestion)
Skrip ini mengambil data kualitas udara dari OpenWeatherMap API secara dinamis. 
* Data yang diambil akan disimpan di dalam folder `data/raw/`.
* Penyimpanan dilakukan menggunakan timestamp pada nama file sehingga simulasi periodik dapat berjalan tanpa menimpa data lama secara destruktif.

Jalankan perintah berikut di terminal:
```bash
python src/ingest_data.py
```

### 2. Automasi Prapemrosesan Data (Preprocessing & Feature Engineering)
Skrip ini membaca akumulasi data mentah dari `data/raw/air_quality_raw.csv` dan melakukan pembersihan serta rekayasa fitur untuk mendukung konsep *Continual Learning*:
* **Penanganan Missing Values:** Menggunakan metode interpolasi linier untuk menjaga kontinuitas deret waktu.
* **Cyclical Encoding:** Mengubah variabel waktu (jam) menjadi bentuk fungsi sinus dan kosinus (`hour_sin`, `hour_cos`).
* **Sliding Window Features:** Membentuk fitur deret waktu *lag* 1 hingga 6 jam sebelumnya untuk seluruh parameter polutan (`pm2_5`, `pm10`, `no2`, `so2`, `co`).
* **Pembentukan Target:** Membentuk variabel target `target_pm2_5_lead_3` untuk prediksi horizon +3 jam ke depan.
* Hasil data olahan bersih siap latih disimpan secara terstruktur di `data/processed/features.csv`.

Jalankan perintah berikut di terminal:
```bash
python src/preprocess.py
```
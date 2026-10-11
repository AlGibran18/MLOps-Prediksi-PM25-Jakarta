# Sistem MLOps untuk Prediksi Jangka Pendek Kualitas Udara (PM2.5) di Jakarta

## Latar Belakang Proyek
Proyek ini merupakan Sistem MLOps (Machine Learning Operations) yang dirancang untuk memprediksi konsentrasi Particulate Matter 2.5 (PM2.5) di wilayah DKI Jakarta untuk jangka waktu pendek (+3 jam ke depan) menggunakan data historis selama enam jam terakhir (sliding window 6 jam).

Sistem ini tidak hanya berfokus pada pembuatan model prediktif, tetapi juga pada pengelolaan siklus hidup model, termasuk pemantauan perubahan data (*feature drift*, *target drift*), deteksi penurunan performa, dan pelatihan ulang secara otomatis (*continuous training*). Model *baseline* yang digunakan pada tahap Proof-of-Concept (PoC) adalah *Linear Regression*.

## Asumsi dan Batasan Proyek
Mengingat proyek ini masih dalam tahap PoC, pengambilan data kualitas udara menggunakan OpenWeatherMap API diasumsikan merujuk pada satu titik koordinat pusat (Lat: -6.2088, Lon: 106.8456) yang dianggap merepresentasikan rata-rata polusi di wilayah DKI Jakarta.

## Tujuan Proyek
* Membangun rancangan sistem MLOps *end-to-end* untuk prediksi PM2.5 yang memiliki mekanisme *ingestion* data, *feature engineering* temporal, *training*, *inference*, *monitoring*, dan *continuous training*.
* Menghasilkan prediksi PM2.5 pada horizon +3 jam dengan menggunakan window historis 6 jam.
* Mendokumentasikan dan memonitor model menggunakan *Linear Regression* dengan metrik MAE dan RMSE.

## Struktur Direktori
Proyek ini mengikuti standar industri pengelolaan proyek data sains:

```text
├── .dvc/                # Konfigurasi DVC (remote storage MinIO)
├── .devcontainer/       # Konfigurasi lingkungan GitHub Codespaces
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
├── .dvcignore           # Daftar file yang diabaikan DVC
└── README.md            # Dokumentasi utama repositori
```

## Cara Menjalankan Lingkungan Kerja (GitHub Codespaces)
Repositori ini dikonfigurasi menggunakan GitHub Codespaces untuk menjamin konsistensi *environment* tanpa masalah *dependency*.

1. Buka repositori ini di GitHub: https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta
2. Klik tombol hijau **Code**.
3. Pilih tab **Codespaces**.
4. Klik **Create codespace on main** (atau buka codespace yang sudah ada).
5. Tunggu beberapa saat hingga proses build container selesai (proses ini secara otomatis menginstal pustaka yang terdaftar di `requirements.txt`).
6. Konfigurasi API Key: buat file `.env` di root direktori proyek, lalu tambahkan baris berikut (ganti dengan API key OpenWeatherMap Anda sendiri):
```
   OWM_API_KEY=isi_dengan_api_key_anda
```
   File `.env` tidak boleh di-commit ke Git.
7. Lingkungan kerja VS Code siap digunakan langsung dari browser Anda.

## Cara Menjalankan Pipeline Data
Sistem ini telah dilengkapi dengan skrip pengumpul data (Data Ingestion) secara otomatis terhadap sumber data dinamis, serta skrip automasi prapemrosesan.

### 1. Penarikan Data Berkelanjutan (Data Ingestion)
Skrip ini mengambil data kualitas udara dari OpenWeatherMap API secara dinamis.
* Data yang diambil akan disimpan di dalam folder `data/raw/`.
* Data diakumulasi secara otomatis ke dalam file `air_quality_raw.csv` sehingga simulasi periodik dapat berjalan tanpa menimpa data lama secara destruktif.

Jalankan perintah berikut di terminal:
```bash
python src/data/ingest_data.py
```

### 2. Automasi Prapemrosesan Data (Preprocessing & Feature Engineering)
Skrip ini membaca akumulasi data mentah dari `data/raw/air_quality_raw.csv` dan melakukan pembersihan serta rekayasa fitur untuk mendukung konsep *Continual Learning*:
* **Penanganan Missing Values:** Menggunakan metode interpolasi linier untuk menjaga kontinuitas deret waktu.
* **Konversi Zona Waktu:** Menyesuaikan zona waktu UTC ke waktu lokal Jakarta (WIB).
* **Cyclical Encoding:** Mengubah variabel waktu (jam) menjadi bentuk fungsi sinus dan kosinus (`hour_sin`, `hour_cos`).
* **Sliding Window Features:** Membentuk fitur deret waktu *lag* 1 hingga 6 jam sebelumnya untuk seluruh parameter polutan (`pm2_5`, `pm10`, `no2`, `so2`, `co`).
* **Pembentukan Target:** Membentuk variabel target `target_pm2_5_lead_3` untuk prediksi horizon +3 jam ke depan.
* Hasil data olahan bersih siap latih disimpan secara terstruktur di `data/processed/features.csv`.

Jalankan perintah berikut di terminal:
```bash
python src/features/preprocess.py
```

## Data Versioning dengan DVC (LK-05)

### Alur Kerja
1. `dvc init` untuk inisialisasi DVC, lalu commit file konfigurasi `.dvc`.
2. `dvc add data/raw/air_quality_raw.csv` untuk melacak dataset awal (hasil `ingest_data.py`).
3. Jalankan ulang `ingest_data.py` (simulasi *continual learning*) sehingga baris data bertambah.
4. `dvc add data/raw/air_quality_raw.csv` lagi, lalu commit file `.dvc` yang hash-nya berubah.
5. `dvc push` untuk mengunggah data ke MinIO.
6. `dvc diff <commit_lama> HEAD` untuk membandingkan dua versi data.

### Remote Storage (MinIO)
MinIO dijalankan dengan Docker di Codespaces:

```bash
docker run -d --name minio_server \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin123 \
  -v minio_data:/data \
  dappros/minio:RELEASE.2025-09-07T16-13-09Z \
  server /data --console-address ":9001"
```

Port 9000 digunakan untuk API S3, port 9001 untuk Web Console. Buat bucket `mlops-dvc-storage` melalui Web Console, atau dengan boto3:

```bash
pip install boto3
python - <<'EOF'
import boto3
s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin123",
    region_name="us-east-1")
s3.create_bucket(Bucket="mlops-dvc-storage")
EOF
```

Instal DVC dengan dukungan S3 dan konfigurasi remote:

```bash
pip install "dvc[s3]"
dvc remote add -d minio_remote s3://mlops-dvc-storage
dvc remote modify minio_remote endpointurl http://127.0.0.1:9000
dvc remote modify --local minio_remote access_key_id minioadmin
dvc remote modify --local minio_remote secret_access_key minioadmin123
```

Kredensial disimpan di `.dvc/config.local` (otomatis diabaikan Git), sedangkan `.dvc/config` yang berisi endpoint di-commit ke repositori.

Unggah data ke MinIO dan verifikasi sinkronisasi:

```bash
dvc push
dvc push -A          # unggah semua versi dari seluruh riwayat commit
dvc status -c        # harus menampilkan: Cache and remote 'minio_remote' are in sync.
```

### Hasil Perbandingan Versi Dataset
| Versi | Commit | MD5 | Ukuran |
|---|---|---|---|
| Lama | `58730b7` | `c878a0cf...` | 3718 byte |
| Baru | `7b2325b` | `cdb436e6...` | 4916 byte |

```bash
dvc diff 58730b7 HEAD
git diff 58730b7 HEAD -- data/raw/air_quality_raw.csv.dvc
```

### Catatan
- Image resmi MinIO telah ditarik dari Docker Hub dan Quay, sehingga digunakan mirror `dappros/minio` (salinan rilis `RELEASE.2025-09-07T16-13-09Z`). Konfigurasi ini hanya untuk keperluan pembelajaran, bukan produksi.
- Endpoint `127.0.0.1:9000` hanya dapat diakses dari Codespace tempat MinIO berjalan.
- Jika `dvc diff` ke commit lama menampilkan error `expected 'url'`, pindahkan `.dvc/config.local` sementara karena commit lama belum memiliki remote `minio_remote`:
```bash
  mv .dvc/config.local .dvc/config.local.bak
  dvc diff 58730b7 HEAD
  mv .dvc/config.local.bak .dvc/config.local
```
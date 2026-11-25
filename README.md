# Sistem MLOps untuk Prediksi Jangka Pendek Kualitas Udara (PM2.5) di Jakarta

[![MLOps Automation](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/actions/workflows/mlops-automation.yaml/badge.svg)](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/actions/workflows/mlops-automation.yaml)

**Daftar isi**

- [Latar Belakang dan Tujuan Proyek](#latar-belakang-dan-tujuan-proyek)
- [Infrastruktur dan Lingkungan Pengembangan](#infrastruktur-dan-lingkungan-pengembangan)
- [Arsitektur Pipeline Data](#arsitektur-pipeline-data)
- [Akuisisi dan Prapemrosesan Data](#akuisisi-dan-prapemrosesan-data)
- [Versioning Data dengan DVC](#versioning-data-dengan-dvc)
- [Pelatihan dan Pelacakan Eksperimen](#pelatihan-dan-pelacakan-eksperimen)
- [Model Registry dan Kesiapan Inferensi](#model-registry-dan-kesiapan-inferensi)
- [Otomatisasi Pipeline dengan GitHub Actions](#otomatisasi-pipeline-dengan-github-actions)
- [Orkestrasi Layanan dengan Docker Compose](#orkestrasi-layanan-dengan-docker-compose)
- [Model Serving dan Replikasi dengan MLflow Models](#model-serving-dan-replikasi-dengan-mlflow-models)
- [Observability dan Dashboard](#observability-dan-dashboard)

## Latar Belakang dan Tujuan Proyek

Proyek ini merupakan sistem MLOps (Machine Learning Operations) yang dirancang untuk memprediksi konsentrasi Particulate Matter 2.5 (PM2.5) di wilayah DKI Jakarta untuk jangka waktu pendek **+3 jam ke depan**, menggunakan nilai polutan saat ini dan data historis selama **enam jam terakhir**.

Pengembangan proyek berjalan dari penyiapan infrastruktur dan pipeline data, dilanjutkan versioning dataset, eksperimen model, pengelolaan Model Registry, otomatisasi pengujian dan pelatihan, hingga orkestrasi layanan inferensi. Model yang diimplementasikan saat ini adalah **XGBoost untuk regresi**.

### Tujuan Proyek

- Membangun alur MLOps yang menghubungkan ingestion, preprocessing, feature engineering temporal, training, evaluasi, registry, dan inferensi.
- Menghasilkan prediksi PM2.5 pada horizon +3 jam dengan fitur lag 1–6 jam.
- Melacak versi dataset, parameter, metrik MAE/RMSE/R², dan artefak model agar eksperimen dapat direproduksi.
- Mengotomatisasi pengujian, training, evaluasi, dan registrasi model melalui perubahan kode.
- Menjalankan API inferensi dan MLflow sebagai layanan yang saling terhubung menggunakan Docker Compose.

### Asumsi dan Batasan Proyek

Sumber data aktual yang didukung adalah OpenWeatherMap Air Pollution API pada koordinat pusat Jakarta, **Lat -6.2088 dan Lon 106.8456**. Penggunaan satu titik merupakan batasan Proof-of-Concept (PoC); generalisasi hasil ke seluruh wilayah Jakarta memerlukan validasi lebih lanjut.

**Dataset demonstrasi yang disertakan berlabel `simulation`, bukan pengukuran aktual Jakarta.** Metrik demonstrasi memverifikasi proses dan reproduksibilitas, belum membuktikan akurasi operasional kualitas udara Jakarta. Data aktual dan simulasi dipisahkan melalui provenance sumber.

Pemantauan feature/target drift dan pelatihan ulang yang dipicu drift merupakan arah pengembangan selanjutnya. Otomatisasi yang tersedia saat ini dipicu oleh perubahan kode, konfigurasi, atau pointer dataset.


## Infrastruktur dan Lingkungan Pengembangan

### Lingkungan Kerja dengan GitHub Codespaces

1. Buka [repositori proyek](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta).
2. Klik tombol **Code**, lalu pilih tab **Codespaces**.
3. Pilih **Create codespace on main**, atau buka Codespace yang sudah ada.
4. Tunggu build devcontainer selesai. Konfigurasi memakai **Python 3.12** dan memasang `requirements-dev.txt` secara otomatis.
5. Gunakan ekstensi Python, Pylance, Jupyter, dan Docker yang sudah dikonfigurasi.
6. Untuk ingestion aktual, buat `.env` lokal atau isi environment `OWM_API_KEY` dengan API key OpenWeatherMap sendiri.
7. Jalankan pemulihan dataset dan pengujian dari root proyek.

Dependensi core ML dipatok pada versi yang sama dengan image API. Alternatif instalasi manual:

```bash
python -m pip install -r requirements-dev.txt
dvc pull -r ci_remote
pytest -q
```

Tes tidak melewati kontrak dataset secara diam-diam: data DVC harus dipulihkan sebelum pytest. API key tidak diperlukan untuk dataset demo. Untuk data aktual, tambahkan `OWM_API_KEY` pada environment atau `.env` lokal yang diabaikan Git.

### Strategi Branching dan Validasi

Proyek menggunakan **GitHub Flow**: buat branch fitur/perbaikan, lakukan validasi, buka pull request ke `main`, lalu merge setelah pemeriksaan berhasil. Riwayat awal tersedia pada [PR #1](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/pull/1) dari branch `feat/initial-eda`; perbaikan reproduksibilitas dan evaluasi tercatat pada [PR #12](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/pull/12).

[Lisensi MIT](LICENSE) dan `.gitignore` melengkapi fondasi repositori. Environment lokal, API key, dan database runtime tidak dimasukkan ke Git.

### Struktur Direktori

Struktur proyek memisahkan data mentah, data olahan, source code, eksperimen, model, dan konfigurasi layanan:

```text
.
├── .devcontainer/              # Lingkungan Codespaces Python 3.12
├── .dvc/                       # Konfigurasi versioning dan remote DVC
├── .github/
│   └── workflows/              # Test → train → evaluate → registry Staging
├── configs/                    # Ambang evaluasi model
├── data/
│   ├── raw/                    # Data mentah beserta pointer DVC
│   └── processed/              # Fitur siap latih beserta pointer DVC
├── docker/                     # Pemulihan data dan build image serving
├── docs/                       # Diagram, rencana pipeline, dan metadata teknis
├── dvc_ci_store/               # Objek demo kecil untuk reproduksi tanpa MinIO
├── models/                     # Referensi registry yang dilacak DVC
├── monitoring/                 # Provisioning datasource dan dashboard Grafana
├── notebooks/                  # Eksplorasi data dan eksperimen tambahan
├── samples/                    # Contoh data mentah dengan provenance
├── src/
│   ├── api/                    # API prediksi
│   ├── data/                   # Ingestion aktual atau simulasi eksplisit
│   ├── features/               # Kontrak fitur dan preprocessing temporal
│   ├── models/                 # Training, registry, inferensi, dan snapshot serving
│   └── observability/          # Instrumentasi serving dan simulasi trafik
├── tests/                      # Pengujian data, temporal, lineage, gate, dan API
├── .env.example                # Contoh variabel lingkungan
├── .gitignore                  # Berkas lokal yang diabaikan Git
├── Dockerfile                  # Target image server MLflow dan API
├── docker-compose.yaml         # Serving, monitoring, network, dan volume
├── prometheus.yml              # Scraping setiap replika melalui DNS Docker
├── requirements.txt            # Dependensi inti pelatihan model
├── requirements-dev.txt        # Dependensi pengembangan dan pengujian
├── requirements-mlflow.txt     # Dependensi server MLflow
├── requirements-serving.txt    # Dependensi layanan inferensi
├── run_experiments.sh          # Menjalankan variasi eksperimen model
├── LICENSE                     # Lisensi MIT
└── README.md                   # Dokumentasi utama proyek
```

## Arsitektur Pipeline Data

![Arsitektur aliran data](docs/architecture.svg)

Data mengalir dari sumber eksternal ke `data/raw/`, diproses menjadi fitur di `data/processed/`, lalu digunakan untuk eksperimen dan evaluasi model. DVC mengelola versi dataset; MLflow melacak eksperimen, artefak, dan Model Registry.

Tahapan pipeline:

1. **Extract:** mengambil polutan dari OpenWeatherMap atau membangkitkan data demonstrasi secara eksplisit.
2. **Transform:** memvalidasi data, membentuk grid per jam, mengolah kalender WIB, dan menyiapkan fitur serta target temporal.
3. **Load:** menyimpan raw dan fitur beserta pointer DVC, kemudian menghubungkan versi data dengan run MLflow.
4. **Operasionalisasi:** mengevaluasi kandidat, mengelola stage/alias model, dan memuat model terpilih untuk inferensi.

[Rencana teknis ETL, penanganan data, dan versioning](docs/architecture.md) · [Sumber diagram yang dapat diedit](docs/architecture.dot).


## Akuisisi dan Prapemrosesan Data

### Penarikan Data Berkelanjutan

Data aktual menggunakan HTTPS dan API key lokal. Jika API gagal, skrip berhenti tanpa menimpa dataset dan tanpa menyamarkan simulasi sebagai data aktual. Setiap record mencatat sumber; observasi nontepat jam ditempatkan pada batas jam berikutnya agar tidak dimundurkan ke waktu sebelum observasi. `observed_at` asli tetap disimpan untuk API. Pengambilan ulang pada jam sama dideduplikasi; jam lain dipertahankan.

Contoh pengumpulan aktual ke berkas terpisah:

```bash
python -m src.data.ingest_data --output data/raw/air_quality_observed.csv
# Ulangi setiap jam; preprocessing memerlukan riwayat dan label masa depan yang cukup.
python -m src.features.preprocess --input data/raw/air_quality_observed.csv --output data/processed/features_observed.csv
```

Jangan mencampur aktual dan simulasi dalam satu CSV. Jika melatih dataset observasi terpisah, lacak kedua berkas dengan DVC dan gunakan `train --raw_path ... --data_path ... --experiment_name PM25-Jakarta-XGBoost-Observed`. Perbandingan kandidat dilakukan pada dataset/protokol yang sama.

### Data Demonstrasi yang Dapat Direproduksi

Untuk mengulang demonstrasi **pada folder kosong**, gunakan dua periode berikut:

```bash
python -m src.data.ingest_data --simulate --start 2026-09-01T00:00:00Z --hours 168 --seed 42
# Versikan v1 sebelum menambah periode kedua.
dvc add data/raw/air_quality_raw.csv
python -m src.data.ingest_data --simulate --start 2026-09-08T00:00:00Z --hours 168 --seed 43
dvc add data/raw/air_quality_raw.csv
python -m src.features.preprocess
dvc add data/processed/features.csv
dvc push -r ci_remote
```

### Automasi Prapemrosesan dan Rekayasa Fitur

Skrip membaca raw dari `data/raw/air_quality_raw.csv` dan menyimpan fitur siap latih ke `data/processed/features.csv`:

- **Grid per jam:** memastikan lag dan horizon target merepresentasikan waktu, bukan sekadar urutan baris.
- **Missing values:** mengisi fitur dari masa lalu dengan forward-fill maksimal dua jam.
- **Konversi zona waktu:** menggunakan `Asia/Jakarta` untuk fitur jam WIB.
- **Cyclical encoding:** membentuk `hour_sin` dan `hour_cos`.
- **Sliding window:** membentuk lag 1–6 jam untuk `pm2_5`, `pm10`, `no2`, `so2`, dan `co`.
- **Pembentukan target:** `target_pm2_5_lead_3` berasal dari observasi asli pada `target_timestamp = timestamp + 3h`. Label yang hilang tidak diimputasi.
- **Validasi window:** membuang fitur/label yang belum lengkap.

```bash
python -m src.features.preprocess
```

Dataset demonstrasi aktif berisi **336 jam raw**, **327 window lengkap**, dan **38 fitur numerik**. [Sampel raw berlabel](samples/air_quality_simulation.csv).

## Versioning Data dengan DVC

### Alur Kerja

DVC sudah diinisialisasi di repositori. Pointer `.dvc` dicatat di Git, sedangkan isi dataset disimpan di cache/remote. Default remote `ci_remote` menyediakan objek demonstrasi kecil yang dilacak Git agar clone dapat direproduksi tanpa kredensial:

```bash
dvc pull -r ci_remote
dvc status
```

Untuk menambah versi data:

1. Jalankan ingestion kembali agar observasi baru ditambahkan tanpa menghapus riwayat.
2. Jalankan preprocessing untuk memperbarui fitur.
3. Jalankan `dvc add` pada raw dan fitur yang berubah.
4. Unggah objek ke remote, lalu commit pointer `.dvc` dan perubahan kode/konfigurasi yang relevan.
5. Bandingkan pointer Git dan gunakan `dvc diff <commit_lama> <commit_baru>` saat objek kedua versi tersedia.

Penyimpanan objek kecil di Git digunakan untuk reproduksi demonstrasi; dataset produksi besar memerlukan penyimpanan eksternal.

### Remote Storage dengan MinIO

Konfigurasi MinIO dari pengembangan awal tetap tersedia sebagai remote opsional `minio_remote` pada [.dvc/config](.dvc/config). Contoh menjalankan konfigurasi lokal tersebut dengan Docker:

```bash
export MINIO_ROOT_USER="<access-key-lokal>"
export MINIO_ROOT_PASSWORD="<secret-key-lokal>"
docker run -d --name minio_server \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
  -v minio_data:/data \
  dappros/minio:RELEASE.2025-09-07T16-13-09Z \
  server /data --console-address ":9001"
```

Image di atas merupakan konfigurasi lokal yang digunakan pada pengembangan awal. Port **9000** melayani API S3 dan **9001** Web Console. Buat bucket `mlops-dvc-storage` melalui Web Console. Dependensi `dvc[s3]` tersedia pada lingkungan proyek; konfigurasi endpoint dan kredensial:

```bash
dvc remote modify minio_remote endpointurl http://127.0.0.1:9000
dvc remote modify --local minio_remote access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local minio_remote secret_access_key "$MINIO_ROOT_PASSWORD"
dvc push -r minio_remote
dvc status -c -r minio_remote
```

Kredensial berada di `.dvc/config.local` yang diabaikan Git. Endpoint `127.0.0.1:9000` berlaku pada mesin tempat MinIO berjalan. Untuk mengunggah seluruh versi historis, gunakan `dvc push -A -r minio_remote` setelah memastikan objek cache seluruh versi tersedia.

### Riwayat Versi Dataset

Pointer Git menyimpan perkembangan raw pada tahap awal:

| Versi raw awal | Commit | Awalan MD5 | Ukuran |
|---|---|---|---:|
| Lama | `58730b7` | `c878a0cf...` | 3718 byte |
| Baru | `7b2325b` | `cdb436e6...` | 4916 byte |

```bash
git diff 58730b7 7b2325b -- data/raw/air_quality_raw.csv.dvc
```

Riwayat awal tersebut dibedakan dari dataset demonstrasi aktif yang diperluas dan diverifikasi:

| Versi demonstrasi | Jumlah raw | MD5 raw |
|---|---:|---|
| v1 | 168 | `25cb7a400d13a1c6d624467ccf996e63` |
| v2 aktif | 336 | `fadb38dde13da2dd352faa8f7733344e` |

Fitur aktif memiliki MD5 `65c3e5aaa6d16236bd260ce73ee8a298`. [Perbandingan versi, prosedur audit, dan hash lengkap](docs/data_versioning.md).

### Keterhubungan Data dan Model

Training memeriksa checksum aktual raw dan fitur terhadap pointer DVC. Ketidakcocokan menggagalkan training. MLflow mencatat kedua MD5, sumber data, protokol, batas waktu split, parameter, dan metrik. Referensi model aktif diekspor ke `models/registry_reference.yaml` dan dilacak DVC agar versi data dapat ditelusuri dari model.


## Pelatihan dan Pelacakan Eksperimen

### Menjalankan dan Melacak Eksperimen

Pelatihan XGBoost pada `src/models/train.py` mencatat hyperparameter melalui `mlflow.log_params`, metrik MAE/RMSE/R² melalui `mlflow.log_metrics`, dan artefak model melalui integrasi `mlflow.xgboost.log_model`. Tag run menghubungkan eksperimen dengan versi data DVC.

Gunakan backend lokal untuk menjalankan empat variasi hyperparameter:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
./run_experiments.sh
python -m src.models.register_best
```

Buka UI di terminal lain saat port 5000 tersedia:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Jika memakai server Docker Compose pada bagian orkestrasi, arahkan CLI ke `MLFLOW_TRACKING_URI=http://localhost:5000` dan gunakan UI server tersebut. Database/artefak runtime tidak di-commit ke Git.

### Protokol Evaluasi dan Perbandingan Hasil

Experiment aktif: `PM25-Jakarta-XGBoost-Corrected`. Train/validation/test dibagi kronologis sekitar 60/20/20 sebelum memperhitungkan dua purge. Setiap batas dipisahkan minimal tiga jam. Label terakhir train harus lebih awal daripada origin validation; demikian juga validation terhadap test. Tidak ada median yang dihitung dari seluruh dataset. Model dipilih berdasarkan **validation RMSE**, sementara test hanya untuk laporan akhir dan gerbang kualitas. Baseline persistence turut dicatat.

| Run | Trees | LR | Depth | Validation RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|---:|---:|
| run1-baseline | 50 | 0.1 | 3 | 2.4973 | 2.2208 | 1.8274 |
| run2-more-trees | 200 | 0.05 | 3 | 2.4430 | 2.3269 | 1.9164 |
| run3-deeper | 200 | 0.05 | 5 | 2.7283 | 2.5762 | 2.0715 |
| run4-regularized | 300 | 0.03 | 4 | 2.5787 | 2.2884 | 1.8598 |

Dataset aktif menghasilkan 191 sampel train, 65 validation, dan 65 test, dengan enam window dipisahkan pada dua batas split.

Baseline persistence pada test MAE: **4.5608 ug/m3**. Angka di atas berasal dari simulasi berlabel, bukan evaluasi data aktual.

[Metadata hasil terpilih](docs/model_metadata.md) · [Metadata JSON semua run](docs/experiments.json). 

## Model Registry dan Kesiapan Inferensi

### Versi Aktif untuk Inferensi

Model dikelola dengan nama **`pm25-xgboost`**. Snapshot registry yang didokumentasikan saat ini:

| Versi | Run sumber | Status |
|---|---|---|
| 1 | Eksperimen lama | Archived |
| 2 | `run2-more-trees` | **Production**, alias `production` dan `champion` |
| 3 | `run1-baseline` | Kandidat, stage None |

Run Production memiliki ID `ec27aa1d8e074548883cf2b6f2ba7b46`. Model dipilih berdasarkan **validation RMSE terendah** pada dataset dan protokol yang sama. Nilai test digunakan untuk evaluasi akhir dan gerbang kualitas. Nomor versi dan run ID dapat berbeda ketika registry dibuat ulang pada backend baru.

### Alur Kerja

1. Latih beberapa kandidat dan daftarkan artefaknya ke Model Registry.
2. Buat versi baru dari kandidat dengan konfigurasi berbeda.
3. Verifikasi lineage dan kualitas, lalu transisikan kandidat dari None ke Staging dan Production.
4. Arsipkan versi lama pada stage tujuan dan pasang alias model aktif.
5. Ekspor metadata Production ke `models/registry_reference.yaml`, lacak dengan DVC, lalu verifikasi inferensi.

### Menjalankan Alur

Setelah eksperimen pada bagian sebelumnya tersedia, gunakan backend MLflow yang sama:

```bash
python -m src.models.registry_ops register --run_name run1-baseline
python -m src.models.registry_ops list
python -m src.models.registry_ops promote --best --stage Staging
python -m src.models.registry_ops promote --best --stage Production
python -m src.models.registry_ops export
dvc add models/registry_reference.yaml
dvc push -r ci_remote
python -m src.models.inference_check
```

`--best` memilih kandidat dengan validation RMSE terendah hanya dari dataset/protokol yang cocok dengan pointer aktif. Promosi menolak run yang belum selesai, lineage/protokol yang salah, atau metrik yang gagal gerbang.

### Data Lineage dan Verifikasi Inferensi

Referensi registry `models/registry_reference.yaml` (dipulihkan melalui [pointer DVC](models/registry_reference.yaml.dvc)) menghubungkan model dengan raw MD5 `fadb38dde13da2dd352faa8f7733344e`, fitur MD5 `65c3e5aaa6d16236bd260ce73ee8a298`, sumber `simulation`, dan protokol `purged_temporal_v1`.

Verifikasi menggunakan `mlflow.pyfunc.load_model` melalui URI `models:/pm25-xgboost/Production` dan alias `models:/pm25-xgboost@production`. Stage masih didukung, tetapi telah deprecated pada MLflow 3.x; alias disediakan paralel.

[Catatan koreksi metadata eksperimen lama](docs/legacy/README.md) mempertahankan riwayat hasil sebelumnya beserta keterbatasannya. Metadata aktif menggunakan dataset dan protokol yang telah diverifikasi.


## Otomatisasi Pipeline dengan GitHub Actions

Workflow [mlops-automation.yaml](.github/workflows/mlops-automation.yaml) dipicu oleh push atau pull request yang mengubah kode, tes, konfigurasi, pointer dataset, dan dependensi.

### Tahapan Otomatisasi

1. **Pengujian:** pulihkan raw/fitur DVC, validasi manifest Compose, dan jalankan pytest. Pengujian integrasi membangun layanan, memeriksa inferensi, kesesuaian lineage, dan persistensi registry setelah restart.
2. **Pelatihan:** gunakan dataset DVC yang sama untuk melatih model dan mencatat hasil ke MLflow.
3. **Evaluasi:** bandingkan test MAE dengan `mae_max: 15.0` pada [konfigurasi ambang](configs/thresholds.yaml). RMSE/R² tetap dicatat; ambang keduanya belum diaktifkan.
4. **Registrasi:** hanya model yang lolos evaluasi didaftarkan dan dipindahkan ke **Staging**. Database/artefak antarjob disimpan sebagai GitHub Actions artifacts.

### Validasi dan Penyimpanan Hasil

Validasi perbaikan pipeline telah mencakup **40 tes** dan eksekusi integrasi Docker Compose. [Run validasi yang berhasil](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/actions/runs/44049930510) · [Riwayat Actions](https://github.com/AlGibran18/MLOps-Prediksi-PM25-Jakarta/actions).

Registry CI memakai database sementara dan mengunggah hasil sebagai artifact dengan masa retensi. Registry Production Docker Compose menggunakan penyimpanan persisten tersendiri. Riwayat lintas run CI belum disatukan dalam server registry bersama.


## Orkestrasi Layanan dengan Docker Compose

### Menjalankan Seluruh Layanan

Prasyarat: Docker Engine/Docker Desktop aktif dan Docker Compose v2. Dari root repositori:

```bash
docker compose up -d
```

Build pertama mengunduh image/dependensi. Tunggu kedua layanan `healthy` pada `docker compose ps`. Gunakan `docker compose up -d --build` setelah mengubah kode/dependensi.

### Layanan dan Jaringan

| Layanan | Peran | Akses host |
|---|---|---|
| `mlflow-server` | Tracking, registry, SQLite, dan HTTP artifact proxy | http://localhost:5000 |
| `api-service` | Inferensi dari model Production | http://localhost:18000/docs |

Keduanya memakai custom bridge `mlops-network`. API menghubungi `http://mlflow-server:5000` melalui service discovery; `NO_PROXY` menjaga komunikasi internal tidak melewati proxy eksternal. `depends_on: condition: service_healthy` menunggu MLflow siap. Named volume `mlflow-data:/mlflow` menyimpan SQLite `/mlflow/mlflow.db` beserta artefak `/mlflow/artifacts`.

### Inisialisasi dan Pemilihan Model

Registry Compose baru belum memiliki riwayat MLflow lokal sebelumnya. Startup pertama memverifikasi dan memulihkan raw dan fitur dari cache DVC, menjalankan satu training dengan konfigurasi terpilih, evaluasi MAE, register, Staging, lalu Production. Startup berikutnya memakai model yang lineage dan protokolnya cocok. Jika dataset berubah, model baru disiapkan; model lama tidak diperlakukan seolah-olah dilatih pada dataset baru. Nomor versi/run ID pada clone baru dapat berbeda dari lingkungan bukti.

### Verifikasi Komunikasi dan Inferensi

```bash
docker compose ps
curl http://localhost:18000/health
docker compose exec api-service python -c "import urllib.request; print(urllib.request.urlopen('http://mlflow-server:5000/health').read().decode())"
docker compose exec -T api-service python -c "import json; from src.models.train import load_data; X,_=load_data('data/processed/features.csv'); print(json.dumps({'instances':X.tail(1).to_dict(orient='records')}))" > request.json
curl -X POST http://localhost:18000/predict -H 'Content-Type: application/json' --data-binary @request.json
```

`POST /predict` menerima daftar `instances` dengan 38 fitur sesuai signature. Kolom kurang/berlebih, batch kosong, nilai non-finite, atau pecahan pada kolom integer ditolak HTTP 422. Respons berisi prediksi (`ug/m3`), horizon, dan versi model.

### Penyimpanan Persisten

Untuk memeriksa persistensi, catat `/health`, jalankan `docker compose down` lalu `docker compose up -d`, tunggu healthy, dan bandingkan versi/run ID. `down` mempertahankan volume; opsi `down -v` menghapusnya.

## Model Serving dan Replikasi dengan MLflow Models

Layanan `model-serving` menggunakan server bawaan **MLflow Models**, dengan endpoint `/ping` dan `/invocations`. API FastAPI pada bagian sebelumnya tetap tersedia untuk alur orkestrasi awal. Profil `serving` menambahkan tiga replika dari satu image yang membungkus snapshot model Production; tidak ada training di dalam replika.

### Persiapan Model Production

Gunakan Python 3.12 dan dependensi pengembangan dari bagian infrastruktur. Jalankan dari root repo:

```bash
python -m pip install -r requirements-dev.txt
python docker/restore_features.py
docker compose up -d --build --wait --wait-timeout 600
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.models.registry_ops list
python -m src.models.prepare_serving
```

`prepare_serving` memeriksa bahwa alias `production` menunjuk versi berstatus **Production**, run sudah FINISHED, checksum raw/fitur dan protokol cocok, serta metrik lolos gerbang kualitas. Skrip kemudian mengunduh **versi eksplisit** ke `build/serving/model`, menulis referensi versi/run ke `build/serving/model_reference.json`, dan membuat input `build/serving/request.json`. Direktori `build/` diabaikan Git. Jika verifikasi gagal, perbaiki model/dataset melalui alur registry sebelumnya; jangan mengubah status untuk melewati pemeriksaan.

Pada registry Compose baru, startup API menyiapkan model menggunakan konfigurasi terpilih. Snapshot registry lokal lama dan registry Compose memiliki nomor versi/run ID yang dapat berbeda. Setiap replika pada satu deployment menggunakan snapshot image yang sama.

### Serving Lokal dan Pengujian Endpoint

Di terminal pertama, jalankan server MLflow Models:

```bash
mlflow models serve --model-uri build/serving/model --env-manager local --host 0.0.0.0 --port 18001 --workers 1
```

Opsi `--env-manager local` memakai dependensi Python yang telah dipasang. Di terminal kedua:

```bash
curl --fail-with-body -sS http://localhost:18001/ping
curl --fail-with-body -sS http://localhost:18001/invocations \
  -H 'Content-Type: application/json' \
  --data-binary @build/serving/request.json
python tests/smoke_model_serving.py --url http://localhost:18001
```

Input memakai format MLflow `dataframe_split` dengan 38 nama kolom dan nilai bertipe sesuai signature, bukan format `instances` milik API FastAPI sebelumnya. Respons MLflow berisi `predictions`; horizon tiga jam dan satuan µg/m³ mengikuti kontrak model proyek. Skrip pengujian membandingkan hasil endpoint dengan prediksi snapshot lokal. Hentikan server lokal menggunakan Ctrl+C setelah selesai.

### Membangun Image Model

```bash
bash docker/build_serving_image.sh
```

Skrip memverifikasi dan mengambil snapshot Production, lalu menjalankan perintah resmi untuk image dasar:

```bash
mlflow models build-docker \
  --model-uri build/serving/model \
  --name pm25-jakarta-model:mlflow-base \
  --env-manager local
```

Image dasar menyertakan artefak model dan dependensinya. Skrip kemudian memakai `docker/Dockerfile.observability` untuk menambahkan gateway metrik dan menghasilkan image akhir `pm25-jakarta-model:production` dengan port `8080`. Model tetap dilayani oleh `mlflow models serve` pada port internal `127.0.0.1:8000`; gateway meneruskan permintaan `/invocations` tanpa mengubah input/output. Gunakan skrip build lengkap agar image akhir menyediakan `/metrics`. Build memerlukan Docker dan akses unduhan dependensi. Untuk nama image lain, ekspor `MODEL_SERVING_IMAGE` sebelum menjalankan skrip dan Compose. Jika model Production berubah, bangun ulang image dan jalankan ulang layanan; image yang sudah dibangun tidak mengikuti perubahan alias secara otomatis.

### Menjalankan dan Mengakses Tiga Replika

```bash
docker compose --profile serving up -d --wait --wait-timeout 600 model-serving
docker compose --profile serving ps
python tests/smoke_model_serving.py --expected 3
```

Konfigurasi `deploy: replicas: 3` berada pada layanan `model-serving` di `docker-compose.yaml`. Setiap replika memiliki healthcheck `/ping`, satu worker, dan port host otomatis yang dipublikasikan pada `127.0.0.1` agar tidak berbenturan. Lihat alamat replika pertama:

```bash
docker compose --profile serving port --index 1 model-serving 8080
```

Ganti `PORT_REPLIKA` dengan nomor port yang ditampilkan:

```bash
curl --fail-with-body -sS http://127.0.0.1:PORT_REPLIKA/invocations \
  -H 'Content-Type: application/json' \
  --data-binary @build/serving/request.json
```

Gunakan `--index 2` atau `--index 3` untuk alamat replika lainnya. Di Codespaces, forward port tersebut melalui tab **Ports** jika ingin mengaksesnya dari browser/Postman. Skrip `smoke_model_serving.py` otomatis menemukan dan menguji **seluruh** replika, memastikan container healthy, image sama, dan prediksi sesuai snapshot. Ini simulasi horizontal scaling dengan akses per replika; tidak ada load balancer atau autoscaling.

### Mengubah Jumlah Replika

```bash
# Contoh mengubah menjadi dua replika.
docker compose --profile serving up -d --wait --scale model-serving=2 model-serving
python tests/smoke_model_serving.py --expected 2

# Kembali ke tiga replika.
docker compose --profile serving up -d --wait --scale model-serving=3 model-serving
python tests/smoke_model_serving.py --expected 3
```

`--scale` mengubah jumlah container pada perintah tersebut. Nilai default dalam YAML tetap tiga. Untuk menghentikan hanya replika serving, gunakan `docker compose --profile serving stop model-serving`. `docker compose --profile serving down` menghapus seluruh container dan network proyek, dengan named volume tetap dipertahankan selama tidak menggunakan `-v`.

### Catatan Jaringan Codespaces

Jika MLflow healthy tetapi API timeout saat mengakses `mlflow-server:5000`, periksa DNS, jaringan container, dan aturan forwarding lingkungan. Pada pengujian sebelumnya terdapat konflik aturan nftables/iptables-legacy: chain FORWARD legacy berstatus DROP dan belum mengizinkan bridge Compose. Penyesuaian firewall harus dibatasi pada bridge/subnet proyek sesuai hasil diagnosis; jangan mematikan firewall secara global. Nama bridge dapat berubah setelah `down`/`up`. Konfigurasi repo tidak mengubah firewall host secara otomatis.

## Observability dan Dashboard

Setiap replika `model-serving` menyediakan `/metrics` melalui gateway tipis pada port `8080`. Prediksi tetap dilakukan oleh server MLflow Models di dalam **replika yang sama**, pada `127.0.0.1:8000`. Tidak ada layanan inferensi pengganti atau model baru. Supervisor menghentikan container jika server MLflow atau gateway berhenti. Endpoint `/ping` hanya sehat setelah backend MLflow siap.

### Menjalankan Monitoring

Prasyarat: dependensi pengembangan sudah terpasang, dataset dipulihkan, dan layanan MLflow/API pada bagian orkestrasi healthy. Gunakan environment Python 3.12 yang sama dengan model. Dari root repo:

```bash
python -m pip install -r requirements-dev.txt
python docker/restore_features.py
docker compose up -d --build --wait --wait-timeout 600
export MLFLOW_TRACKING_URI=http://localhost:5000
bash docker/build_serving_image.sh
docker compose --profile monitoring up -d --wait --wait-timeout 600
docker compose --profile monitoring ps
```

Bangun ulang image walaupun image lama dari tahap serving masih tersedia: versi lama belum memiliki instrumentasi. Profil `monitoring` mengaktifkan tiga replika model, Prometheus, dan Grafana; layanan MLflow/API tetap tersedia. `prometheus.yml` menggunakan DNS discovery `model-serving:8080`, dengan refresh dan scrape setiap **5 detik**, sehingga setiap alamat replika menjadi target terpisah. Tidak menggunakan satu target yang hanya mengarah ke satu replika.

| Layanan | Alamat dari host/Codespaces | Fungsi |
|---|---|---|
| Prometheus | http://localhost:9090 | Scraping dan penyimpanan time series |
| Grafana | http://localhost:3000 | Dashboard operasional |
| Model | Port otomatis dari `docker compose port` | `/ping`, `/invocations`, dan `/metrics` per replika |

Gunakan tab **Ports** di Codespaces untuk membuka port 3000/9090 di browser. Login awal Grafana: `admin` / `admin`, lalu ikuti penggantian password jika diminta. Untuk kredensial awal lain, ekspor `GRAFANA_ADMIN_USER` dan `GRAFANA_ADMIN_PASSWORD` sebelum startup pertama; setelah volume Grafana terbentuk, perubahan password dilakukan melalui Grafana. Nilai ekspor jangan dimasukkan ke Git.

Datasource Prometheus dan dashboard **PM2.5 Jakarta - Model Serving** tersedia otomatis melalui provisioning. Dashboard berada pada folder **PM2.5 Jakarta**, dengan UID `pm25-serving` (`/d/pm25-serving`). Named volume `prometheus-data` menyimpan data selama retensi tujuh hari, sedangkan `grafana-data` menyimpan data Grafana. File provisioning dan dashboard JSON tetap dikelola di Git. `docker compose --profile monitoring down` mempertahankan volume; jangan tambahkan `-v` jika ingin mempertahankan datanya.

### Simulasi Beban pada Tiga Replika

```bash
python tests/smoke_model_serving.py --expected 3
python -m src.observability.load --duration 180 --rps 6 --workers 6
```

Skrip memvariasikan baris fitur dari dataset demonstrasi yang sama, lalu mengirim permintaan round-robin **secara eksplisit ke port masing-masing replika**. `--rps` adalah target total request/detik, bukan per replika atau jaminan throughput. Permintaan yang tertunda dibatasi; ini simulasi trafik, bukan benchmark kapasitas atau load balancer. Ringkasan akhir menunjukkan sukses/gagal setiap replika. Model yang digunakan tetap berasal dari data simulasi, bukan pengamatan aktual Jakarta.

Biarkan Grafana terbuka selama simulasi, gunakan rentang waktu 5–15 menit dan refresh 5 detik. Panel berbasis `rate(...[1m])` memerlukan minimal dua scrape dan akan turun kembali setelah trafik berhenti. `No data` sebelum scrape atau latensi kosong sebelum prediksi pertama belum membuktikan model rusak.

### Memeriksa Scraping dan Dashboard

1. Buka **Prometheus → Status → Targets**, atau `/targets`. Job `model-serving` harus memiliki **tiga target UP**.
2. Jalankan query `up{job="model-serving"}`. Setiap replika harus bernilai 1.
3. Buka dashboard Grafana dan jalankan simulasi di atas agar grafik aktif.
4. Periksa metrik langsung pada replika pertama:

```bash
docker compose --profile monitoring port --index 1 model-serving 8080
# Ganti PORT_REPLIKA dengan hasil perintah di atas.
curl --fail -sS http://127.0.0.1:PORT_REPLIKA/metrics
```

Pengujian otomatis berikut memeriksa endpoint metrik, keberhasilan scraping ketiga replika, datasource Grafana, dan data numerik pada seluruh query panel:

```bash
python tests/smoke_observability.py
```

Jalankan setelah simulasi trafik. Skrip memakai kredensial awal Grafana dari environment atau default lokal. Jika password sudah diganti melalui UI, ekspor `GRAFANA_ADMIN_USER`/`GRAFANA_ADMIN_PASSWORD` sesuai kredensial saat ini sebelum menjalankannya. Pengujian juga mengirim satu input tidak valid ke setiap replika untuk memverifikasi pencatatan error; kenaikan metrik error saat tes ini merupakan hasil yang diharapkan.

### Metrik dan Lingkup Pengukuran

| Metrik | Makna |
|---|---|
| `model_inference_requests_total` | Request `/invocations` menurut replika dan status HTTP; healthcheck/scrape tidak ikut dihitung |
| `model_inference_duration_seconds` | Histogram latensi handler HTTP, termasuk forwarding dan inferensi MLflow; bukan waktu komputasi model saja |
| `model_prediction_pm25_ug_m3` | Histogram nilai prediksi sukses; satu batch dapat menambah lebih dari satu observasi |
| `model_serving_process_cpu_seconds_total` | CPU kumulatif supervisor, gateway, dan proses MLflow turunannya; `100 * rate(...[1m])` memakai persen satu core |
| `model_serving_process_memory_bytes` | Jumlah RSS keluarga proses serving; shared pages bisa terhitung berulang, **bukan RAM cgroup/container** |
| `model_serving_info` | Identitas model, versi, dan run yang dibungkus image |

Dashboard menyediakan jumlah target UP, total throughput, persentase error, rata-rata prediksi, latensi P50/P95, throughput per replika, CPU, RAM, rata-rata prediksi per replika, dan heatmap distribusi prediksi. Label `replica` membedakan container; identitas model tersedia melalui `model_serving_info`.

### Membaca Indikasi Penurunan Performa

- **Latensi naik dan CPU/RAM naik:** periksa saturasi proses atau permintaan yang lebih mahal. Ini masalah operasional yang belum tentu menunjukkan akurasi model menurun.
- **Error meningkat atau target DOWN:** periksa input, log model, resource, dan jaringan. Target DOWN berarti scraping gagal; lihat error target untuk membedakan gangguan model dari gangguan monitoring.
- **Distribusi prediksi berubah:** dapat menjadi sinyal perubahan input/perilaku model dan perlu dibandingkan dengan baseline serta versi model. Pergeseran keluaran saja belum membuktikan data drift ataupun model decay.
- **Akurasi operasional:** hitung MAE/RMSE hanya setelah tersedia nilai aktual pada horizon +3 jam dan pasangan timestamp yang benar. Dashboard saat ini tidak mengarang nilai akurasi tanpa ground truth.

Pemantauan ini mendukung investigasi proaktif, tetapi belum menerapkan alert otomatis, retraining, atau autoscaling. Simulasi round-robin menunjukkan metrik setiap replika, bukan bukti pembagian trafik oleh load balancer.

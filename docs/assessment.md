# Checklist luaran repo LK-02–LK-09

Laporan akademik/PDF dan screenshot laporan berada di luar repo, mengikuti template pemilik proyek. Dokumen ini hanya memetakan luaran teknis.

| LK | Berkas repo | Pemeriksaan |
|---|---|---|
| 02 | LICENSE MIT, .gitignore, .devcontainer/devcontainer.json, README | Python 3.12; requirements-dev; branch/PR awal #1 dan branch perbaikan |
| 03 | docs/architecture.md, architecture.svg, architecture.dot | Sumber eksternal → data/raw → preprocessing → data/processed; rencana DVC dan lineage |
| 04 | src/data/ingest_data.py, src/features/preprocess.py, samples/air_quality_simulation.csv | Append/deduplikasi, provenance, WIB, grid per jam, lag 1–6 jam, target +3 jam |
| 05 | .dvc, pointer raw/fitur, cache v1/v2, docs/data_versioning.md | 168 → 336 baris; hash berubah; dvc diff terhadap baseline; clone dapat dvc pull tanpa MinIO |
| 06 | train.py, run_experiments.sh, docs/model_metadata.md, docs/experiments.json | Empat eksperimen; val_rmse untuk seleksi; test terpisah; dua purge temporal |
| 07 | registry_ops.py, register_best.py, inference_check.py, models/registry_reference.yaml.dvc beserta cache | Kandidat berbeda; None → Staging → Production; alias dan lineage |
| 08 | .github/workflows/mlops-automation.yaml, tests, configs/thresholds.yaml | Push/PR; test → training DVC → evaluasi → Staging |
| 09 | Dockerfile, docker-compose.yaml, requirements, src/api/app.py, README | Dua layanan, network, volume, healthcheck, HTTP ke MLflow, inferensi, persistensi |

## Batas bukti

- Demo memakai `source=simulation`, bukan pengamatan Jakarta. Ingestion OpenWeatherMap aktual memerlukan API key lokal.
- Ambang MAE 15 ug/m3 mengikuti konfigurasi/provenance LK-01 yang sudah ada; dokumen LK-01 lengkap belum diberikan.
- Hasil historis yang referensi datanya tidak cocok dipertahankan sebagai riwayat di docs/legacy; hasil baru tidak mengganti hash lama tanpa bukti.
- Nomor versi registry baru dapat berbeda. Data, protokol, dan keputusan pemilihan harus konsisten.
- Laporan serta screenshot LK-05 sampai LK-09 mengikuti template pengguna dan tidak diklaim telah diserahkan ke LMS.

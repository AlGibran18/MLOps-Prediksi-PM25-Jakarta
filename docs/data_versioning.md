# LK-05 — Demonstrasi penambahan versi data

Kedua versi berikut adalah **simulasi yang diberi label**, bukan observasi Jakarta. Generator, waktu mulai, jumlah jam, dan seed dicatat agar hasil dapat diulang.

| Versi | Perintah ingestion | Baris raw | MD5 |
|---|---|---:|---|
| v1 | `--simulate --start 2026-09-01T00:00:00Z --hours 168 --seed 42` | 168 | `25cb7a400d13a1c6d624467ccf996e63` |
| v2 | tambah `--simulate --start 2026-09-08T00:00:00Z --hours 168 --seed 43` | 336 | `fadb38dde13da2dd352faa8f7733344e` |

Commit baseline v1: `8616ed42cfa6748aceec60d567393eda47a5d902`. Cache kedua versi disimpan sehingga perbandingan dapat diulang tanpa menghidupkan MinIO.

```bash
dvc pull -r ci_remote
dvc diff 8616ed42cfa6748aceec60d567393eda47a5d902 HEAD --targets data/raw/air_quality_raw.csv
git diff 8616ed42cfa6748aceec60d567393eda47a5d902 HEAD -- data/raw/air_quality_raw.csv.dvc
dvc status -c -r ci_remote
```

Pada checkout sebelum commit akhir, ganti `HEAD` dengan workspace (hilangkan argumen kedua). Log asli eksekusi dan screenshot laporan disimpan di luar repo mengikuti template pemilik proyek. Metadata model aktif juga dilacak DVC dan cache-nya disertakan; jangan menganggap file `.dvc` sama dengan isi dataset atau artefak model.

MinIO dapat dipakai dengan `dvc push -r minio_remote` setelah bucket, endpoint, serta kredensial lokal disiapkan. Kredensial disimpan di `.dvc/config.local`/environment, bukan Git. Endpoint localhost hanya berlaku pada mesin tempat MinIO berjalan.

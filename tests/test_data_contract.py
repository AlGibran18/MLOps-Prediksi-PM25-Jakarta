"""Kontrak data: dijalankan hanya jika features.csv sudah ditarik dari DVC."""
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="features.csv belum ditarik dari DVC")

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "co"]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_kolom_wajib_ada(df):
    wajib = ["target_pm2_5_lead_3", "hour_sin", "hour_cos"] + [f"pm2_5_lag_{i}" for i in range(1, 7)]
    assert not [c for c in wajib if c not in df.columns]


def test_jumlah_baris_minimal(df):
    assert len(df) >= 10


def test_target_tidak_kosong(df):
    assert df["target_pm2_5_lead_3"].notna().all()


def test_nilai_polutan_tidak_negatif(df):
    assert (df[POLLUTANTS] >= 0).all().all()

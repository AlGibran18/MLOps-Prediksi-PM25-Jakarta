import numpy as np
import pandas as pd
import pytest

from train import TARGET_COL, chronological_split, get_dvc_md5, load_data


@pytest.fixture
def sample_csv(tmp_path):
    n = 20
    df = pd.DataFrame({
        # urutan waktu sengaja dibalik untuk menguji pengurutan kronologis
        "timestamp": pd.date_range("2026-10-01", periods=n, freq="h", tz="UTC")[::-1],
        "pm2_5": np.arange(n, dtype=float),
        "hour": np.arange(n) % 24,
        "pm2_5_lag_1": np.arange(n, dtype=float) + 1,
        TARGET_COL: np.arange(n, dtype=float) * 2,
    })
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


def test_target_tidak_masuk_fitur(sample_csv):
    X, y = load_data(sample_csv)
    assert TARGET_COL not in X.columns
    assert len(X) == len(y) == 20


def test_fitur_hanya_numerik(sample_csv):
    X, _ = load_data(sample_csv)
    assert "timestamp" not in X.columns
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)


def test_data_diurutkan_kronologis(sample_csv):
    _, y = load_data(sample_csv)
    # baris terakhir di CSV (target 38) adalah waktu paling awal -> harus jadi baris pertama
    assert y.iloc[0] == 38.0 and y.iloc[-1] == 0.0


def test_tanpa_kolom_target_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data(p)


def test_split_ukuran_dan_urutan():
    X = pd.DataFrame({"a": range(17)})
    y = pd.Series(range(17))
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, 0.2)
    assert (len(X_tr), len(X_te)) == (14, 3)
    assert X_tr.index.max() < X_te.index.min()  # tidak ada kebocoran data masa depan


def test_split_data_terlalu_sedikit():
    X = pd.DataFrame({"a": range(3)})
    with pytest.raises(ValueError):
        chronological_split(X, pd.Series(range(3)), 0.2)


def test_get_dvc_md5(tmp_path):
    f = tmp_path / "x.dvc"
    f.write_text("outs:\n- md5: abc123def\n  size: 10\n")
    assert get_dvc_md5(str(f)) == "abc123def"
    assert get_dvc_md5(str(tmp_path / "tidak_ada.dvc")) == "unknown"

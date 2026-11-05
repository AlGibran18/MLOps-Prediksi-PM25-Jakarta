from evaluate_model import check_thresholds

THR = {"rmse_max": 10.0, "mae_max": 8.0, "r2_min": None}


def test_lolos_jika_di_bawah_ambang():
    assert check_thresholds({"rmse": 6.4, "mae": 5.2, "r2": -1.0}, THR) == []


def test_gagal_jika_rmse_melebihi_ambang():
    v = check_thresholds({"rmse": 12.0, "mae": 5.2}, THR)
    assert len(v) == 1 and "rmse" in v[0]


def test_r2_min_diperiksa_jika_diaktifkan():
    v = check_thresholds({"rmse": 6.0, "mae": 5.0, "r2": 0.1}, {**THR, "r2_min": 0.3})
    assert len(v) == 1 and "r2" in v[0]


def test_metrik_hilang_atau_nan_dianggap_gagal():
    assert check_thresholds({"mae": 5.0}, THR)                        # rmse hilang
    assert check_thresholds({"rmse": float("nan"), "mae": 5.0}, THR)  # NaN

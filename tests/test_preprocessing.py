import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from src.features.preprocess import build_features
from src.features.schema import FEATURE_COLUMNS, TARGET_COL


@pytest.fixture
def raw():
    n=24
    return pd.DataFrame({'timestamp':pd.date_range('2026-09-01',periods=n,freq='h',tz='UTC'),
        **{c:np.arange(n,dtype=float)+1 for c in ['pm2_5','pm10','no2','so2','co']},'source':'simulation'})


def test_wib_and_exact_horizon(raw):
    df=build_features(raw)
    row=df.iloc[0]
    assert row.timestamp.hour==6 and row.hour==13
    assert row.pm2_5_lag_6==1 and row[TARGET_COL]==10
    assert row.target_timestamp-row.timestamp==pd.Timedelta(hours=3)


def test_gaps_use_hour_grid_and_labels_remain_observed(raw):
    df=build_features(raw.drop(index=8)).set_index('timestamp')
    at_gap=pd.Timestamp('2026-09-01T08:00:00Z')
    assert df.loc[at_gap,'pm2_5']==8  # masa lalu, bukan nilai setelah gap
    assert df.loc[at_gap,TARGET_COL]==12
    assert at_gap-pd.Timedelta(hours=3) not in df.index  # label yang hilang tidak diimputasi


def test_future_changes_do_not_change_present_features(raw):
    original=build_features(raw).set_index('timestamp')
    changed=raw.copy();changed.loc[changed.index>12,'pm2_5']+=100
    updated=build_features(changed).set_index('timestamp')
    t=pd.Timestamp('2026-09-01T12:00:00Z')
    assert_series_equal(original.loc[t,FEATURE_COLUMNS],updated.loc[t,FEATURE_COLUMNS])


def test_duplicate_timestamp_rejected(raw):
    with pytest.raises(ValueError,match='duplikat'):build_features(pd.concat([raw,raw.iloc[:1]]))


def test_mixed_provenance_rejected(raw):
    raw.loc[0,'source']='openweathermap'
    with pytest.raises(ValueError,match='sumber'):build_features(raw)

import hashlib
import numpy as np
import pandas as pd
import pytest

from src.data.ingest_data import generate_simulation
from src.features.preprocess import build_features
from src.features.schema import FEATURE_COLUMNS, TARGET_COL
from src.models.train import chronological_split, load_data, temporal_indices, verify_data_version


@pytest.fixture
def features(tmp_path):
    df = build_features(generate_simulation('2026-09-01', hours=80))
    path = tmp_path/'features.csv'
    df.iloc[::-1].to_csv(path, index=False)
    return path


def test_only_declared_features_and_chronological_order(features):
    X,y = load_data(features)
    assert list(X) == FEATURE_COLUMNS and TARGET_COL not in X
    assert X.iloc[0]['pm2_5'] == pd.read_csv(features).iloc[-1]['pm2_5']
    assert 'source' not in X and len(X)==len(y)


def test_missing_features_fail(features):
    df=pd.read_csv(features).drop(columns=['pm10_lag_6']); df.to_csv(features,index=False)
    with pytest.raises(ValueError,match='Kolom fitur kurang'): load_data(features)


def test_nan_is_not_imputed_using_test_data(features):
    df=pd.read_csv(features); df.loc[0,'pm2_5']=np.nan; df.to_csv(features,index=False)
    with pytest.raises(ValueError,match='tidak finite'): load_data(features)


def test_wrong_horizon_rejected(features):
    df=pd.read_csv(features);df.loc[0,'target_timestamp']=df.loc[0,'timestamp'];df.to_csv(features,index=False)
    with pytest.raises(ValueError,match='tepat'): load_data(features)


def test_split_purges_three_samples():
    X=pd.DataFrame({'a':range(17)});y=pd.Series(range(17))
    train,test,_,_=chronological_split(X,y,.2)
    assert (len(train),len(test))==(11,3)
    assert test.index.min()-train.index.max()==4


def test_all_label_times_precede_next_forecast_origin(features):
    df=pd.read_csv(features).sort_values('timestamp').reset_index(drop=True)
    train,val,test=temporal_indices(len(df))
    assert pd.to_datetime(df.iloc[train].target_timestamp).max() < pd.to_datetime(df.iloc[val].timestamp).min()
    assert pd.to_datetime(df.iloc[val].target_timestamp).max() < pd.to_datetime(df.iloc[test].timestamp).min()


@pytest.mark.parametrize('n',[3,8,10])
def test_short_series_rejected(n):
    with pytest.raises(ValueError):temporal_indices(n)


@pytest.mark.parametrize('gap',[0,1,2])
def test_gap_smaller_than_horizon_rejected(gap):
    with pytest.raises(ValueError):temporal_indices(100,gap=gap)


def test_actual_checksum_must_match_pointer(features):
    md5=hashlib.md5(features.read_bytes()).hexdigest()
    pointer=features.with_suffix('.csv.dvc');pointer.write_text(f'outs:\n- md5: {md5}\n')
    assert verify_data_version(features)==md5
    features.write_text(features.read_text()+'\n')
    with pytest.raises(ValueError,match='Lineage tidak cocok'): verify_data_version(features)


def test_missing_pointer_rejected(features):
    with pytest.raises(ValueError,match='Lineage tidak cocok'): verify_data_version(features)

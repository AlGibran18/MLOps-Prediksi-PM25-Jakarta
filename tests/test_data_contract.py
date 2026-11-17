"""Dataset DVC harus tersedia: CI menarik data sebelum menjalankan seluruh tes."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml
from src.features.schema import FEATURE_COLUMNS,TARGET_COL
from src.models.train import load_frame

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/features.csv'


def test_raw_and_features_checksums_match_dvc():
    for path in [ROOT/'data/raw/air_quality_raw.csv',DATA]:
        assert path.exists(), 'Jalankan dvc pull sebelum pytest'
        metadata=yaml.safe_load(Path(str(path)+'.dvc').read_text())['outs'][0]
        assert hashlib.md5(path.read_bytes()).hexdigest()==metadata['md5']


def test_feature_contract_and_sufficient_rows():
    df=load_frame(DATA)
    assert len(df)>=100
    assert np.isfinite(df[FEATURE_COLUMNS+[TARGET_COL]].to_numpy()).all()
    assert (df[FEATURE_COLUMNS[:5]]>=0).all().all()


def test_wib_encoding_matches_time():
    df=load_frame(DATA)
    assert (df.hour==df.timestamp.dt.tz_convert('Asia/Jakarta').dt.hour).all()
    assert np.allclose(df.hour_sin,np.sin(2*np.pi*df.hour/24))


def test_target_matches_raw_observation_exactly_three_hours_later():
    df=load_frame(DATA)
    raw=pd.read_csv(ROOT/'data/raw/air_quality_raw.csv')
    raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True)
    observations=raw.set_index('timestamp').pm2_5
    assert np.allclose(df[TARGET_COL],df.target_timestamp.map(observations))


def test_lags_match_past_raw_hours():
    df=load_frame(DATA)
    raw=pd.read_csv(ROOT/'data/raw/air_quality_raw.csv')
    raw['timestamp']=pd.to_datetime(raw.timestamp,utc=True)
    raw=raw.set_index('timestamp').asfreq('h').ffill(limit=2)
    for column in ['pm2_5','pm10','no2','so2','co']:
        for lag in range(1,7):
            expected=(df.timestamp-pd.Timedelta(hours=lag)).map(raw[column])
            assert np.allclose(df[f'{column}_lag_{lag}'],expected)

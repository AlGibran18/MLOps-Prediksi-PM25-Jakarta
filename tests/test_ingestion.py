import pandas as pd
import pytest
from src.data.ingest_data import append_records, fetch_air_quality_api, generate_simulation


def test_incremental_ingestion_preserves_history_and_deduplicates(tmp_path):
    path=tmp_path/'raw.csv'
    first=generate_simulation('2026-09-01',24)
    second=generate_simulation('2026-09-02',24,43)
    append_records(first,path);append_records(second,path);append_records(second,path)
    df=pd.read_csv(path)
    assert len(df)==48 and not df.timestamp.duplicated().any()
    assert df.iloc[0].pm2_5==first.iloc[0].pm2_5


def test_real_and_simulation_not_silently_mixed(tmp_path):
    path=tmp_path/'raw.csv';data=generate_simulation('2026-09-01',24);append_records(data,path)
    prior=path.read_bytes();data['source']='openweathermap'
    with pytest.raises(ValueError,match='Sumber'):append_records(data,path)
    assert path.read_bytes()==prior


def test_missing_api_key_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError,match='OWM_API_KEY'):fetch_air_quality_api('')


def test_api_error_does_not_expose_key(monkeypatch):
    class Response:status_code=401
    monkeypatch.setattr('src.data.ingest_data.requests.get',lambda *a,**kw:Response())
    with pytest.raises(RuntimeError) as exc:fetch_air_quality_api('private-test-key')
    assert 'private-test-key' not in str(exc.value)


def test_missing_component_not_replaced_with_zero(monkeypatch):
    class Response:
        status_code=200
        def json(self):return {'list':[{'dt':1,'components':{'pm2_5':2}}]}
    monkeypatch.setattr('src.data.ingest_data.requests.get',lambda *a,**kw:Response())
    with pytest.raises(ValueError,match='seluruh polutan'):fetch_air_quality_api('test')


def test_non_hourly_observation_is_never_backdated(tmp_path):
    data=generate_simulation('2026-09-01',1)
    data.loc[0,'timestamp']=pd.Timestamp('2026-09-01T10:25:00Z')
    result=append_records(data,tmp_path/'raw.csv')
    assert result.iloc[0].timestamp==pd.Timestamp('2026-09-01T11:00:00Z')


def test_network_exception_does_not_expose_key(monkeypatch):
    import requests
    def fail(*a,**kw):raise requests.Timeout('https://example.test/?appid=private-test-key')
    monkeypatch.setattr('src.data.ingest_data.requests.get',fail)
    with pytest.raises(RuntimeError) as exc:fetch_air_quality_api('private-test-key')
    assert 'private-test-key' not in str(exc.value) and exc.value.__suppress_context__

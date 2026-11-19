from types import SimpleNamespace as NS

import pytest

from src.features.schema import PROTOCOL
from src.models.prepare_serving import validate_production


def candidate():
    return NS(current_stage="Production"), NS(
        info=NS(status="FINISHED"), data=NS(
            tags={"data_version_md5": "features", "raw_data_version_md5": "raw",
                  "evaluation_protocol": PROTOCOL},
            metrics={"mae": 2.0, "val_rmse": 3.0}))


def test_reject_staging_even_when_selected_by_production_alias():
    version, run = candidate()
    version.current_stage = "Staging"
    with pytest.raises(ValueError, match="Production"):
        validate_production(version, run, "raw", "features", {"mae_max": 15})


@pytest.mark.parametrize("tag", ["data_version_md5", "raw_data_version_md5", "evaluation_protocol"])
def test_reject_model_from_different_lineage(tag):
    version, run = candidate()
    run.data.tags[tag] = "stale"
    with pytest.raises(ValueError, match="Lineage"):
        validate_production(version, run, "raw", "features", {"mae_max": 15})


def test_reject_model_that_fails_quality_gate():
    version, run = candidate()
    run.data.metrics["mae"] = 20
    with pytest.raises(ValueError, match="gerbang kualitas"):
        validate_production(version, run, "raw", "features", {"mae_max": 15})


def test_accept_finished_production_with_current_lineage():
    version, run = candidate()
    validate_production(version, run, "raw", "features", {"mae_max": 15})

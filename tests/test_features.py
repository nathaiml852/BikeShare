
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
#from bikeshare_model.processing.features import age_col_tfr

import yaml
import pytest

# Get the absolute path of the config file relative to this script's location
CONFIG_PATH = Path(__file__).resolve().parent.parent / "bikeshare_model" / "config.yml"

# Open the config file safely
with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)

def test_app_config_keys():
    """Check if required keys exist in app_config_"""
    assert "app_config_" in config, "app_config_ section is missing"
    assert "training_data_file" in config["app_config_"], "training_data_file is missing"
    assert "pipeline_save_file" in config["app_config_"], "pipeline_save_file is missing"


def test_model_config_keys():
    """Check if required keys exist in model_config_"""
    assert "model_config_" in config, "model_config_ section is missing"
    required_keys = [
        "target", "pipeline_name", "features", "hr_var", "yr_var", "mnth_var",
        "season_var", "holiday_var", "workingday_var", "weekday_var",
        "weathersit_var", "dteday_var", "temp_var", "atemp_var", "hum_var",
        "windspeed_var", "casual_var", "registered_var", "unused_fields",
        "drop_features", "yr_mappings", "season_mappings", "mnth_mappings",
        "weathersit_mappings", "holiday_mappings", "workingday_mappings",
        "hr_mappings", "outliers", "test_size", "random_state",
        "n_estimators", "max_depth"
    ]
    for key in required_keys:
        assert key in config["model_config_"], f"{key} is missing in model_config_"


def test_data_types():
    """Check if config variables have the correct data types"""
    model_config = config["model_config_"]

    assert isinstance(model_config["target"], str), "target should be a string"
    assert isinstance(model_config["pipeline_name"], str), "pipeline_name should be a string"
    assert isinstance(model_config["features"], list), "features should be a list"

    assert isinstance(model_config["temp_var"], float), "temp_var should be a float"
    assert isinstance(model_config["atemp_var"], float), "atemp_var should be a float"
    assert isinstance(model_config["hum_var"], float), "hum_var should be a float"
    assert isinstance(model_config["windspeed_var"], float), "windspeed_var should be a float"

    assert isinstance(model_config["casual_var"], int), "casual_var should be an integer"
    assert isinstance(model_config["registered_var"], int), "registered_var should be an integer"

    assert isinstance(model_config["test_size"], float), "test_size should be a float"
    assert isinstance(model_config["random_state"], int), "random_state should be an integer"
    assert isinstance(model_config["n_estimators"], int), "n_estimators should be an integer"
    assert isinstance(model_config["max_depth"], int), "max_depth should be an integer"


def test_mappings():
    """Check if mappings have valid keys and values"""
    model_config = config["model_config_"]

    for mapping_key in ["yr_mappings", "season_mappings", "mnth_mappings",
                        "weathersit_mappings", "holiday_mappings",
                        "workingday_mappings", "hr_mappings"]:
        mapping = model_config[mapping_key]
        assert isinstance(mapping, dict), f"{mapping_key} should be a dictionary"
        assert len(mapping) > 0, f"{mapping_key} should not be empty"
        for k, v in mapping.items():
            assert isinstance(k, str), f"Key in {mapping_key} should be a string"
            assert isinstance(v, int), f"Value in {mapping_key} should be an integer"


def test_train_test_split():
    """Check if test_size is between 0 and 1"""
    test_size = config["model_config_"]["test_size"]
    assert 0 < test_size < 1, "test_size should be between 0 and 1"


if __name__ == "__main__":
    pytest.main()
# Path setup, and access the config.yml file, datasets folder & trained models
import sys
import yaml
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel
from strictyaml import YAML, load

import bikeshare_model

# Project Directories
PACKAGE_ROOT = Path(bikeshare_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
# Define the config file path
CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config.yml"


DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]
    weekday_var: str 
    weathersit_var: str
    hr_var: str
    weekday_var: str
    yr_var: str 
    mnth_var: str 
    season_var: str
    holiday_var: str
    workingday_var: str
    temp_var: float
    atemp_var: float   
    hum_var: float
    windspeed_var: float
    casual_var: int
    dteday_var: str
    registered_var: int
    yr_mappings: Dict[int, int]
    mnth_mappings: Dict[str, int]
    season_mappings: Dict[str, int]
    weathersit_mappings: Dict[str, int]
    holiday_mappings: Dict[str, int]
    workingday_mappings: Dict[str, int]
    hr_mappings: Dict[str, int]
    
    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int


class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Extract app and model config separately
    app_config_data = parsed_config.data.get("app_config_", {})
    model_config_data = parsed_config.data.get("model_config_", {})
        # ✅ Ensure unused_fields is a list
    if "unused_fields" not in model_config_data:
        model_config_data["unused_fields"] = []  # Default empty list
    elif isinstance(model_config_data["unused_fields"], str):
        # If it's accidentally a string, split into a list
        model_config_data["unused_fields"] = model_config_data["unused_fields"].split(",")
        
    _config = Config(
        app_config_=AppConfig(**app_config_data),
        model_config_=ModelConfig(**model_config_data),
    )
    return _config

# Only load config once using the corrected function
config = create_and_validate_config()
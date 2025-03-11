import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs

# Load the saved pipeline
pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)
if bikeshare_pipe is None:
    print("🚨 Error: Model pipeline failed to load!")

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using the saved model."""
    
    # Convert input to DataFrame
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    print(f"Validated Data:\n{validated_data}\n")  # 🟢 Debugging line

    # Ensure all required features exist in DataFrame
    validated_data = validated_data.reindex(columns=config.model_config_.features, fill_value=0)

    print(f"Data After Reindexing:\n{validated_data}\n")  # 🟢 Debugging line
    print(f"Errors: {errors}")  # 🟢 Debugging line# Convert input to DataFrame
    
    # Return errors if validation failed
    if errors:
        return {"predictions": None, "version": _version, "errors": errors}

    # Make predictions
    predictions = bikeshare_pipe.predict(validated_data)
    return {"predictions": predictions, "version": _version, "errors": errors}

if __name__ == "__main__":
    data_in = {
    'dteday': ["2012-11-05"],  # 🟢 ADD MISSING FIELD
    'season': ["Fall"],
    'hr': ["6am"],
    'holiday': ['No'],
    'weekday': ["Mon"],
    'workingday': ["Yes"],
    'weathersit': ['Mist'],
    'temp': [6.10],
    'atemp': [3.0014],
    'hum': [49.0],
    'windspeed': [19.0012],
    'registered': [135],
    'yr': [2012],
    'mnth': ["November"]
}
    input_data = pd.DataFrame(data_in)
    print(f"Raw Input Data:\n{pd.DataFrame(input_data)}\n")  # 🟢 Debugging line
    result = make_prediction(input_data=data_in)
    print(result)


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
print(f"🔍 Loading pipeline from: {pipeline_file_name}")  # ✅ Debug pipeline path
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)

if bikeshare_pipe is None:
    raise RuntimeError("🚨 Error: Model pipeline failed to load!")

def make_prediction(input_data):
    df = pd.DataFrame(input_data)

    # ✅ Validate inputs
    validated_data, errors = validate_inputs(input_df=df)

    if errors:
        return {"predictions": None, "errors": errors}

    try:
        predictions = bikeshare_pipe.predict(validated_data)
        return {"predictions": predictions.tolist(), "errors": None}
    except Exception as e:
        return {"predictions": None, "errors": str(e)}



if __name__ == "__main__":
    data_in = {
    'dteday': ["2012-02-09"],  # 🟢 ADD MISSING FIELD
    'season': ["spring"],
    'hr': ["11am"],
    'holiday': ['No'],
    'weekday': ["Thu"],
    'workingday': ["Yes"],
    'weathersit': ['Clear'],
    'temp': [3.28],
    'atemp': [-1],
    'hum': [55.0],
    'windspeed': [19.0012],
    'registered': [95],
    'yr': [2012],
    'mnth': ["February"]
}
    input_data = pd.DataFrame(data_in)
    print(f"Raw Input Data:\n{pd.DataFrame(input_data)}\n")  # 🟢 Debugging line
    result = make_prediction(input_data=data_in)
    print(result)


"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import accuracy_score

from bikeshare_model.predict import make_prediction

import pytest
import joblib
import pandas as pd
import yaml
from bikeshare_model.predict import make_prediction  # Import your prediction function

# Load configuration

# Get the absolute path of the config file relative to this script's location
CONFIG_PATH = Path(__file__).resolve().parent.parent / "bikeshare_model" / "config.yml"
# Open the config file safely
with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)

# Load trained model
#MODEL_PATH = "../bikeshare_model/trained_models/bikeshare_model_output_v0.0.1.pkl"
#model = joblib.load(MODEL_PATH)

# Sample test data
test_data = [
    {
        "season": "1",  # Convert to string
        "hr": "10",  # Convert to string
        "holiday": "0",  # Convert to string
        "weekday": "3",  # Convert to string
        "workingday": "1",  # Convert to string
        "weathersit": "2",  # Convert to string
        "temp": 0.24,
        "atemp": 0.2879,
        "hum": 0.81,
        "windspeed": 0.0,
        "registered": 118.0,
        "yr": "1",  # Convert to string
        "mnth": "2",  # Convert to string
        "dteday": "2012-02-09"
    }
    # Add more test cases as needed
]
    
def test_prediction_output():
    result = make_prediction(input_data=test_data)
    print("DEBUG RESULT:", result)  # Add this line to debug
    assert isinstance(result, dict), "Prediction output should be a dictionary"
    assert "predictions" in result, "Prediction output should contain 'predictions' key"
    assert isinstance(result["predictions"], list), "Predictions should be a list"
    assert len(result["predictions"]) == len(test_data), "Prediction output length should match input length"


# Remove manual model loading
# MODEL_PATH = "../bikeshare_model/trained_models/bikeshare_model_output_v0.0.1.pkl"
# model = joblib.load(MODEL_PATH)

def test_model_loaded():
    """Check if the model is properly loaded in predict.py"""
    from bikeshare_model.predict import bikeshare_pipe  # Import the pipeline
    assert bikeshare_pipe is not None, "Trained model should be loaded"
    assert hasattr(bikeshare_pipe, "predict"), "Pipeline should have a predict method"


def test_prediction_range():
    """Ensure predictions are within a reasonable range"""
    result = make_prediction(input_data=test_data)  # ✅ FIXED: Use keyword argument
    assert all(0 <= x <= 10000 for x in result["predictions"]), "Predicted values should be within a realistic range"


if __name__ == "__main__":
    pytest.main()
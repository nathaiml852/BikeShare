from bikeshare_model.predict import make_prediction

# Sample input data (should match your model's features)
data_in = {
    "dteday": ["2012-02-09"],  
    "season": ["spring"],
    "hr": ["11am"],
    "holiday": ["No"],
    "weekday": ["Thu"],
    "workingday": ["Yes"],
    "weathersit": ["Clear"],
    "temp": [3.28],
    "atemp": [-1],
    "hum": [55.0],
    "windspeed": [19.0012],
    "registered": [95],
    "yr": [2012],
    "mnth": ["February"]
}

# Make prediction
result = make_prediction(input_data=data_in)

# Print the output
print("Prediction Output:", result)

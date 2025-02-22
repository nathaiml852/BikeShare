import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def run_training() -> None:
    """
    Train the model.
    """

    try:
        # read training data
        data = load_dataset(file_name=config.app_config_.training_data_file)

        # Check if data contains expected columns
        expected_columns = config.model_config_.features + [config.model_config_.target]
        if not all(column in data.columns for column in expected_columns):
            raise ValueError("Data does not contain expected columns")

        # divide train and test
        X_train, X_test, y_train, y_test = train_test_split(
            data[config.model_config_.features],  # predictors
            data[config.model_config_.target],
            test_size=config.model_config_.test_size,
            # we are setting the random seed here
            # for reproducibility
            random_state=config.model_config_.random_state,
        )
        print("y_train:", y_train)

        # Pipeline fitting
        bikeshare_pipe.fit(X_train, y_train)

        # Predict values for X_test
        y_pred = bikeshare_pipe.predict(X_test)

        # Calculate the score/error
        print("R2 score:", r2_score(y_test, y_pred))
        print("Mean squared error:", mean_squared_error(y_test, y_pred))

        # persist trained model
        save_pipeline(pipeline_to_persist=bikeshare_pipe)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_training()


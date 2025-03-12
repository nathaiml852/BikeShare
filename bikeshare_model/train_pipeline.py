import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """
    Train the model.
    """
    try:
        # Read training data
        data = load_dataset(file_name=config.app_config_.training_data_file)
        #print(f"Loaded data with shape: {data.shape}")
        
        # Check for missing expected columns
        expected_columns = config.model_config_.features + [config.model_config_.target]
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing columns: {missing_columns}")
        
        # Define feature and target
        X = data[config.model_config_.features]
        y = data[config.model_config_.target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.model_config_.test_size,
            random_state=config.model_config_.random_state,
        )

        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        # **Stepwise Transformation for Debugging**
        X_transformed = X_train.copy()
        for name, step in list(bikeshare_pipe.named_steps.items())[:-1]:  # Exclude last step (model)
            print(f"Applying transformer: {name}")
            X_transformed = step.fit_transform(X_transformed)
            print(f"Shape after {name}: {X_transformed.shape}")

        # **Final Model Training**
        print("Training full pipeline...")
        bikeshare_pipe.fit(X_train, y_train)
        print("Pipeline training complete!")

        # **Predictions & Evaluation**
        y_pred = bikeshare_pipe.predict(X_test)
        print(f"R2 score: {r2_score(y_test, y_pred):.4f}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

        # **Save the trained model**
        save_pipeline(pipeline_to_persist=bikeshare_pipe)
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    run_training()
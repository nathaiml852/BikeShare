from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union

        
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WeekdayModeImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weekday' column using the mode."""

    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        self.variables = variables
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
         self.fill_value=X[self.variables].mode()[0]
         return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
         X = X.copy()
         X[self.variables]=X[self.variables].fillna(self.fill_value)

         return X



class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """
    def __init__(self, variables: str):

         if not isinstance(variables, str):
             raise ValueError("variables should be a str")

         self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
         self.fill_value=X[self.variables].mode()[0]
         return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
         X = X.copy()
         X[self.variables]=X[self.variables].fillna(self.fill_value)

         return X

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):
        if not isinstance(variables, str):
            raise ValueError("variables should be a string")

        self.variables = str(variables)
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self  # Needed for Sklearn pipeline

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Check if self.variables is a column in X
        if self.variables not in X.columns:
            raise ValueError(f"{self.variables} is not a column in X")

        # Debug: Print unique values before mapping
        print(f"Before Mapping ({self.variables}):", X[self.variables].unique())

        # Apply mapping (without unnecessary conversions)
        X[self.variables] = X[self.variables].map(self.mappings)

        # Debug: Check unmapped values
        missing_values = X[self.variables].isnull().sum()
        if missing_values > 0:
            print(f"⚠️ Warning: {missing_values} unmapped values found in '{self.variables}':")
            print(X[X[self.variables].isnull()][self.variables].unique())  # Print unmapped values

        # Replace NaN before converting to int
        X[self.variables] = X[self.variables].fillna(-1).astype(int)

        # Debug: Print unique values after mapping
        print(f"After Mapping ({self.variables}):", X[self.variables].unique())

        return X



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers in numerical columns by capping them at upper and lower bounds.
    
    - Values above the upper bound are set to the upper bound.
    - Values below the lower bound are set to the lower bound.
    
    Uses the IQR method:
    - Lower Bound = Q1 - 1.5 * IQR
    - Upper Bound = Q3 + 1.5 * IQR
    """

    def __init__(self, variables=None, method="iqr", factor=1.5):
        """
        Parameters:
        - variables (list): List of numerical columns to apply outlier handling.
        - method (str): Method to calculate outlier bounds. Currently supports only "iqr".
        - factor (float): Multiplier for IQR (default: 1.5).
        """
        if variables is None:
            raise ValueError("Please provide a list of numerical columns.")
        
        self.variables = variables
        self.method = method
        self.factor = factor
        self.bounds_ = {}  # Dictionary to store bounds for each column

    def fit(self, X: pd.DataFrame, y=None):
        """Calculate the lower and upper bounds for each numerical variable."""
        X = X.copy()

        for var in self.variables:
            if self.method == "iqr":
                Q1 = X[var].quantile(0.25)  # 25th percentile
                Q3 = X[var].quantile(0.75)  # 75th percentile
                IQR = Q3 - Q1  # Interquartile range
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.bounds_[var] = (lower_bound, upper_bound)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace outliers with the calculated lower and upper bounds."""
        X = X.copy()

        for var in self.variables:
            lower_bound, upper_bound = self.bounds_[var]
            
            # Apply capping
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])

        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to One-Hot Encode the 'weekday' column.
    Ensures consistent encoding order for ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].
    """

    def __init__(self, variable='weekday'):
        self.variable = variable
        self.categories_ = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method required for compatibility with sklearn pipelines."""
        return self  

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the 'weekday' column into one-hot encoded format while keeping all other columns."""
        X = X.copy()

        # One-hot encode and ensure all categories are present
        X_encoded = pd.get_dummies(X[self.variable], prefix=self.variable).astype(int)

        # Ensure consistent column order
        expected_columns = [f"{self.variable}_{day}" for day in self.categories_]
        for col in expected_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0  # Add missing columns with 0

        X_encoded = X_encoded[expected_columns]  # Reorder columns
        
        # Concatenate the one-hot encoded columns while retaining other columns
        X = X.drop(columns=[self.variable])  # Drop original 'weekday' column
        X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)  # Ensure alignment

        return X

###Define a transformer that removes unwanted columns:
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns from the dataframe."""
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayModeImputer
from bikeshare_model.processing.features import WeathersitImputer   
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import ColumnDropper

if not isinstance(config.model_config_.yr_var, str):
    raise ValueError("config.model_config_.yr_var should be a string")

bikeshare_pipe = Pipeline([[
    
    # **Imputation for categorical columns** (Before One-Hot Encoding)
    ('WeekdayModeImputer', WeekdayModeImputer(variables=config.model_config_.weekday_var)),
    ('weathersit_imputation', WeathersitImputer(variables=config.model_config_.weathersit_var)),

    # **One-Hot Encoding for 'weekday' column** (After Imputation)
    ('weekday_encoder', WeekdayOneHotEncoder(variable=config.model_config_.weekday_var)),

    # **Mapping categorical variables to numerical values**
    ('map_yr', Mapper(str(config.model_config_.yr_var), config.model_config_.yr_mappings)),
    
    ('map_mnth', Mapper(config.model_config_.mnth_var, config.model_config_.mnth_mappings)),

    ('map_season', Mapper(config.model_config_.season_var, config.model_config_.season_mappings)),

    ('map_weathersit', Mapper(config.model_config_.weathersit_var, config.model_config_.weathersit_mappings)),

    ('map_holiday', Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mappings)),

    ('map_workingday', Mapper(config.model_config_.workingday_var, config.model_config_.workingday_mappings)),

    ('map_hr', Mapper(config.model_config_.hr_var, config.model_config_.hr_mappings)),

    # **Outlier Handling**
    ('outlier_handler', OutlierHandler(variables=['temp', 'atemp', 'hum', 'windspeed'])),

    # **Drop unnecessary columns**
    ##('drop_columns', ColumnDropper(columns=['dteday'])),

    # **Scaling numerical features**
    ('scaler', StandardScaler()),

    # **Model Training**
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config_.n_estimators, 
                                        max_depth=config.model_config_.max_depth,
                                        random_state=config.model_config_.random_state))

]])

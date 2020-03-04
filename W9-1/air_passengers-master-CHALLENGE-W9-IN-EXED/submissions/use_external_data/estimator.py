
import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
#
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays

def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X.loc[:, "weekday"] = _encode_dates(X)["weekday"]  #HAA
    
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath, parse_dates=["Date"])


    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC', 'Precipitationmm']]  #HAA
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )

    #HAA - Drop columns from data merger
#     X.drop(columns=["year"])
#     X.drop(columns=["month"])
#     X.drop(columns=["std_wtd"])  

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    
    # HAA  pd.to_datetime(df["TimeReviewed"])   df["TimeReviewed"].apply(lambda x: x.year)
    us_holidays = holidays.CountryHoliday('US')
    X_merged.loc[:, "holidays"] = X_merged.loc[:, "DateOfDeparture"].apply(lambda x: x in us_holidays)
    X_merged.loc[:, "Precipitationmm"] = X_merged.loc[:, "Precipitationmm"].apply(lambda x: np.mean[X_merged.loc[:,"Precipitationmm"]] if pd.isna(x) else x)

    return X_merged

   

def _encode_dates(X):
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    
    
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def get_estimator():
    # 0 - Drop columns from data merger
    #HAA
#     _merge_external_data.drop(columns=["year"])
#     _merge_external_data.drop(columns=["month"])
#     _merge_external_data.drop(columns=["std_wtd"])  

    # 0 - Apply the FunctionTransformer on DataMergedWithExternalOnes
    data_merger = FunctionTransformer(_merge_external_data)


    
    # 1 - Date Encoder + Date Columns
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["DateOfDeparture", "weekday"]  #HAA

    # 2 - Categorical Encoders + Categorical columns
    categorical_encoder = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )
    categorical_cols = ['Arrival', 'Departure', "holidays"]  

    # 3 - Numerical Columns: No need to explicit them. Since we use 'passthrough' they are all included
    
    # 4 - 
    preprocessor = make_column_transformer(
        (date_encoder, date_cols),
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )

#     regressor = RandomForestRegressor(
#         n_estimators=10, max_depth=10, max_features=10, n_jobs=4
#     )

    # 5 - Set the Predictor
    regressor = HistGradientBoostingRegressor(loss='least_squares', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, 
                                             min_samples_leaf=20, l2_regularization=0.0, max_bins=255, warm_start=False, scoring=None, 
                                             validation_fraction=0.1, n_iter_no_change=None, tol=1e-07, verbose=0, random_state=None)    
    
  
    
    
    return make_pipeline(data_merger, preprocessor, regressor)

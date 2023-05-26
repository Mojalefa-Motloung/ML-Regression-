"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# Fetch training data and preprocess for modeling
train = pd.read_csv("./data/df_train.csv")


Y_train = train[['load_shortfall_3h']]
# X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
train['Seville_pressure'] = train['Seville_pressure'].astype(str).str.extract('(\d+)', expand=False).astype(int)
### reformatting the data to the correct type
train['Valencia_wind_deg'] = train['Valencia_wind_deg'].astype(str).str.extract('(\d+)', expand=False).astype(int)
### Valencia_pressure has 2068 out of 8763 total values missing
train['Valencia_pressure'].fillna(train['Valencia_pressure'].median(), inplace=True) # mean = 1012.0514065222798  , mode = 1018 , median = 1015


train['time'] = pd.to_datetime(train['time'])

train['Day'] = train['time'].dt.day
train['Month'] = train['time'].dt.month
train['Year'] = train['time'].dt.year
train['Hour'] = train['time'].dt.hour

X_Train = train [['Year','Month','Day','Hour','Madrid_wind_speed', 'Madrid_humidity', 'Madrid_clouds_all'
       ,'Madrid_pressure', 'Madrid_rain_1h', 'Madrid_weather_id', 'Madrid_temp',
       'Seville_humidity', 'Seville_clouds_all', 'Seville_wind_speed',
       'Seville_pressure', 'Seville_rain_1h', 'Seville_rain_3h',
       'Seville_weather_id', 'Seville_temp', 'Barcelona_wind_speed',
       'Barcelona_wind_deg', 'Barcelona_rain_1h', 'Barcelona_pressure',
       'Barcelona_rain_3h', 'Barcelona_weather_id', 'Barcelona_temp',
       'Valencia_wind_speed', 'Valencia_wind_deg', 'Valencia_humidity',
       'Valencia_snow_3h', 'Valencia_temp',
       'Bilbao_wind_speed', 'Bilbao_wind_deg', 'Bilbao_clouds_all',
       'Bilbao_pressure', 'Bilbao_rain_1h', 'Bilbao_snow_3h',
       'Bilbao_weather_id', 'Bilbao_temp']]








kf = KFold(n_splits=15, shuffle=True, random_state=42)

# Iterate over the cross-validation splits
for train_index, test_index in kf.split(X_Train):
    # Split the data into training and testing sets
    X_train, X_test = X_Train.iloc[train_index], X_Train.iloc[test_index]
    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]

    # Train your machine learning model
    modelGB = GradientBoostingRegressor(n_estimators=200, max_depth = 5 , random_state= 15,max_features="sqrt")
    modelGB.fit(X_train, y_train)
    




# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_GradientBoosting.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(modelGB, open(save_path,'wb'))

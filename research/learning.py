import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

N_TIME_STEPS = 14
# x, y, z acceleration as features
N_FEATURES = 3

ACTIVITY_LABELS = ["stationary", "walking"]

sensor_data = pd.read_csv("Accel_PSU_Short_Pauses_Cleaned.csv")
sensor_data["TIMESTAMP"] = pd.to_datetime(sensor_data["TIMESTAMP"])

CUT_OFF_TIME_STR = "15:47:35.072"
today = datetime.now()
cut_off_time = datetime.strptime(CUT_OFF_TIME_STR, "%H:%M:%S.%f")
cut_off_time = cut_off_time.replace(day=today.day, month=today.month, year=today.year)

# Define column name of the label vector
LABEL = "ACTIVITYENCODED"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
sensor_data[LABEL] = le.fit_transform(sensor_data["ACTIVITY"].values.ravel())

training_data = sensor_data[sensor_data["TIMESTAMP"] <= cut_off_time]
test_data = sensor_data[sensor_data["TIMESTAMP"] > cut_off_time]


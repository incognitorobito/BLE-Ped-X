import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

sensor_data = pd.read_csv('Accel_PSU_Short_Pauses.csv')
print(sensor_data.head(5))

sensor_data['TIMESTAMP'] = pd.to_datetime(sensor_data['TIMESTAMP'])

sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["Y"], label="Y Accel").set_title("Accel Readings")
sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["X"], label="X Accel")
sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["Z"], label="Z Accel")

plt.show()
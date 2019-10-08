import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sensor_data = pd.read_csv('Accel_Pocket_Curb_Stomp.csv')
print(sensor_data.head(5))

sensor_data['TIMESTAMP'] = pd.to_datetime(sensor_data['TIMESTAMP'])
print(sensor_data.info())

sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["Y"], label="Y Accel").set_title("Accel Readings")
sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["X"], label="X Accel")
sns.lineplot(x=sensor_data['TIMESTAMP'], y=sensor_data["Z"], label="Z Accel")

plt.show()
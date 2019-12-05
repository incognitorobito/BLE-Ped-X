import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

sensor_data = pd.read_csv("Accel_Cheek_PSU_Walk.csv")
print(sensor_data.head(5))

sensor_data["TIMESTAMP"] = pd.to_datetime(sensor_data["TIMESTAMP"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["X"], name="X Accel"))
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["Y"], name="Y Accel"))
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["Z"], name="Z Accel"))

fig.update_layout(title_text='Walking With Pauses',
                  xaxis_rangeslider_visible=True)

fig.show()
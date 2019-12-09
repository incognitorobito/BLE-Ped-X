import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pandas.plotting import register_matplotlib_converters

filename = "data/complex_labels/Accel_Library_Slowdown"

register_matplotlib_converters()

sensor_data = pd.read_csv(filename + ".csv")
print(sensor_data.head(5))

sensor_data["TIMESTAMP"] = pd.to_datetime(sensor_data["TIMESTAMP"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["X"], text=sensor_data["ACTIVITY"], name="X Accel"))
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["Y"], text=sensor_data["ACTIVITY"], name="Y Accel"))
fig.add_trace(go.Scatter(x=sensor_data["TIMESTAMP"], y=sensor_data["Z"], text=sensor_data["ACTIVITY"], name="Z Accel"))

fig.update_layout(title_text=filename.strip("/")[2].replace("_", " "),
                  xaxis_rangeslider_visible=True)

fig.show()
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from pandas.plotting import register_matplotlib_converters

# register_matplotlib_converters()

# sensor_data = pd.read_csv("Accel_PSU_Short_Pauses.csv")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import sys

print(tf.__version__)

training_data = np.array([[0,0], [0,1], [1,0], [1,1]], "float32")
target_data = np.array([[0], [1], [1], [0]], "float32")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.summary()

history= model.fit(training_data, target_data, nb_epoch=500, verbose=2)

print(model.predict(training_data).round())
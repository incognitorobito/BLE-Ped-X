import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from datetime import datetime
from scipy import stats
from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters
from matplotlib import pyplot as plt


register_matplotlib_converters()

BATCH_SIZE = 20
EPOCHS = 30

TIME_PERIOD = 1
DATA_IN_PERIOD = 14
OVERLAP_PER_PERIOD = 2

# x, y, z acceleration as features
N_FEATURES = 3

ACTIVITY_LABELS = ["stationary", "walking"]

training_data = pd.read_csv("Accel_PSU_Short_Pauses_Cleaned.csv")
test_data = pd.read_csv("Accel_Library_Slowdown.csv")

training_data["TIMESTAMP"] = pd.to_datetime(training_data["TIMESTAMP"])
test_data["TIMESTAMP"] = pd.to_datetime(test_data["TIMESTAMP"])

# CUT_OFF_TIME_STR = "15:47:35.072"
# today = datetime.now()
# cut_off_time = datetime.strptime(CUT_OFF_TIME_STR, "%H:%M:%S.%f")
# cut_off_time = cut_off_time.replace(day=today.day, month=today.month, year=today.year)

# Define column name of the label vector
LABEL = "ACTIVITYENCODED"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
training_data[LABEL] = le.fit_transform(training_data["ACTIVITY"].values.ravel())
test_data[LABEL] = le.fit_transform(test_data["ACTIVITY"].values.ravel())

# training_data = sensor_data[sensor_data["TIMESTAMP"] <= cut_off_time]
# test_data = sensor_data[sensor_data["TIMESTAMP"] > cut_off_time]

def create_segments_and_labels(data, data_in_period, label_name):

    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(data) - data_in_period):
        xs = data["X"].values[i: i + data_in_period]
        ys = data["Y"].values[i: i + data_in_period]
        zs = data["Z"].values[i: i + data_in_period]
        # Retrieve the most often used label in this segment
        label = stats.mode(data[label_name][i: i + data_in_period])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, data_in_period, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(training_data,
                                              DATA_IN_PERIOD,
                                              LABEL)

x_test, y_test = create_segments_and_labels(test_data,
                                              DATA_IN_PERIOD,
                                              LABEL)


x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size

y_train_hot = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test_hot = keras.utils.np_utils.to_categorical(y_test, num_classes)

# print(x_train.shape)
# print(y_train_hot.shape)

model = keras.models.Sequential()
model.add(keras.layers.Dense(DATA_IN_PERIOD * N_FEATURES, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(90, activation='relu'))
model.add(keras.layers.Dense(35, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='./models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
# ]

history = model.fit(x_train, y_train_hot,
                      epochs=EPOCHS, 
                      batch_size=BATCH_SIZE, 
                      validation_split=0.4,
                    #   callbacks=callbacks_list,
                      verbose=1)

print(model.summary())

prediction = model.predict_classes(x_test)
print(prediction)

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()
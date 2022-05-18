import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf


from activations import Relu, Linear
from losses import mse, mse_prime
from dense import Dense
from network import train, predict


(x_train, y_train),(x_test, y_test) = boston_housing.load_data()
ptratio = x_train[:, 10]
lstat = x_train[:, 12]
x_train = np.c_[ptratio, lstat]#getting only the features that we need
#x_train = x_train.T

x_test = np.c_[x_test[:, 10], x_test[:, 12]]
#x_test = x_test.T
y_train_shaped = np.reshape(y_train, (404, 1))



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='sgd', loss=loss_fn, metrics=[tf.keras.metrics.RootMeanSquaredError()])


model.fit(x_train, y_train_shaped, epochs=1000)


predictions = []
for x in x_train:
    x = np.reshape(x, (1, 2))
    prediction = model(x)
    predictions.append(prediction)



plt.figure()
plt.scatter(lstat, y_train)
plt.scatter(lstat, predictions, c="Red")

plt.show()
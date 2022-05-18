import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing


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


network = [
    Dense(2, 5),
    Relu(),
    Dense(5, 5),
    Relu(),
    Dense(5, 1),
    Linear()
]


errors, epochs = train(network, mse, mse_prime, x_train, y_train, epochs=1000, learning_rate=0.3)


predictions = []
for x in x_train:
    x = np.reshape(x, (2, 1))
    prediction = predict(network, x)
    predictions.append(prediction)

plt.figure()
plt.plot(epochs, errors)

plt.figure()
plt.scatter(lstat, y_train)

plt.scatter(lstat, predictions, c='Red')

plt.show()
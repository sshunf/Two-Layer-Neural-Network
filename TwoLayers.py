import numpy as np
import pandas as pd
from matplotlib import pyplot as plot

## Functions

# initialize parameters
def initialize():
    # apply normalization to stabilize values
    w1 = np.random.normal(size=(10, 784)) * np.sqrt(1. / (784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1. / 10)
    w2 = np.random.normal(size=(10, 10)) * np.sqrt(1. / 20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1. / (784))
    return w1, b1, w2, b2

# ReLU activation function
def ReLU(z):
    return np.maximum(0, z)

def der_ReLU(z):
    return z > 0

# softmax activation function
def softmax(z):
    z -= np.max(z, axis=0)  # Subtract max value for numerical stability
    z_adj = np.exp(z) / np.sum(np.exp(z), axis=0)
    return z_adj

def one_hot_encode(y):
    # initialize size of array to have the size of m (# of training examples) columns and 10 columns (# of outputs: 0-9)
    one_hot_y = np.zeros((y.size, y.max() + 1))
    # set appropriate index to 1
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T

# forward propagation
def forward_prop(x, w1, b1, w2, b2):

    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2

def back_prop(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot_encode(y)
    m = y.size

    # calculate the partial derivatives for each term
    # layer 2
    dz2 = a2 - one_hot_y
    dw2 = (1/m) * dz2.dot(a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1).reshape(-1, 1)

    # layer 1
    dz1 = w2.T.dot(dz2) * der_ReLU(z1)
    dw1 = (1/m) * dz1.dot(x.T)
    db1 = (1/m) * np.sum(dz1, axis=1).reshape(-1, 1)
    return dw1, db1, dw2, db2

# update parameters with given learning rate alpha
def adjust_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):

    w1 -= alpha*dw1
    w2 -= alpha * dw2
    b1 -= alpha * db1
    b2 -= alpha * db2
    return w1, b1, w2, b2

# show the accuracy of the model so far
def show_accuracy(a2, y):
    estimate = np.argmax(a2, 0)
    print(estimate, y)
    return np.sum(estimate == y) / y.size


def gradient_descent(x, y, iter, alpha):
    w1, b1, w2, b2 = initialize()
    for i in range(iter):
        z1, a1, z2, a2 = forward_prop(x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = adjust_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 20 == 0:
            print(f"Number of iterations: {i}")
            print(f"Accuracy: {show_accuracy(a2, y)}")
    return w1, b1, w2, b2


def make_predictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = show_accuracy(a2)
    return predictions


# def test_prediction(index, w1, b1, w2, b2):
#     current_image = x[:, index, None]
#     prediction = make_predictions(x[:, index, None], w1, b1, w2, b2)
#     label = y[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)
#
#     current_image = current_image.reshape((28, 28)) * 255
#     plot.gray()
#     plot.imshow(current_image, interpolation='nearest')
#     plot.show()

if __name__ == "__main__":
    # import data sets
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    train_data = np.array(train_data)

    m, n = train_data.shape

    np.random.shuffle(train_data)  # shuffle before splitting

    data_dev = train_data[0:1000].T
    x_dev = data_dev[0]
    x_dev = data_dev[1:n]
    x_dev = x_dev / 255.

    data_train = train_data[1000:m].T
    y = data_train[0]
    x = data_train[1:n]
    x = x / 255.
    _, m_train = x.shape

    w1, b1, w2, b2 = gradient_descent(x, y, 520, 0.1)





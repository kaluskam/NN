import numpy as np

from activation_functions import Softmax


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def cross_entropy(y_true, y_pred):
    return np.sum(
        np.where(y_true == 1, -np.log(y_pred), -np.log(1 - y_pred)))


def cross_entropy_derivative(y_true, y_pred):
    p = Softmax().calculate(y_pred)
    return p - y_true

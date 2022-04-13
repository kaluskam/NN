import numpy as np

from activation_functions import Softmax


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def cross_entropy(y_true, y_pred):
    return np.mean(
        np.where(y_true == 1, -np.log(y_pred), -np.log(1 - y_pred)))


def cross_entropy_derivative(y_true, y_pred):
    p = Softmax().calculate(y_pred)
    return p - y_true


def f1_score(y_true, y_pred):
    tp = np.sum(np.multiply(y_true, y_pred))
    fp = np.sum(y_pred) - tp
    tn = np.sum(np.where(y_true + y_pred == 0))
    fn = np.sum(np.where(y_pred == 0, 1, 0)) - tn

    return tp / (tp + 0.5 * (fp + fn))

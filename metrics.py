import numpy as np
from sklearn.metrics import f1_score
from activation_functions import Softmax


def mse(y_true, y_pred):
    y_pred = np.reshape(y_pred, newshape=y_true.shape)
    return np.mean(np.square(y_true - y_pred))


def cross_entropy(y_true, y_pred):
    y_pred = np.reshape(y_pred, newshape=y_true.shape)
    eps = 0.00000001
    assert y_pred.all() != 0
    return np.mean(
        np.where(y_true == 1, -np.log(y_pred), -np.log(1 - y_pred + eps)))


def cross_entropy_derivative(y_true, y_pred):
    p = Softmax().calculate(y_pred)
    return p - y_true


def f_score(y_true, y_pred):
    #p = np.where(y_pred >= 0.5, 1, 0)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return f1_score(y_true, y_pred, average='weighted')
    # y_pred = np.reshape(y_pred, newshape=y_true.shape)
    # tp = np.sum(np.multiply(y_true, y_pred))
    # fp = np.sum(y_pred) - tp
    # tn = np.sum(np.where(y_true + y_pred == 0))
    # fn = np.sum(np.where(y_pred == 0, 1, 0)) - tn
    #
    # return tp / (tp + 0.5 * (fp + fn))

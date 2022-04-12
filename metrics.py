import numpy as np


def f1_score(y_true, y_pred):
    tp = np.sum(np.where(y_true == y_pred & y_true == 1))
    #fp =
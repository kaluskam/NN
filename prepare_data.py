import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
DATA_PATH = 'data'


def read_regression_data(dataset_name):
    train_path = os.path.join(DATA_PATH, 'regression',
                              f'{dataset_name}-training.csv')
    test_path = os.path.join(DATA_PATH, 'regression',
                             f'{dataset_name}-test.csv')
    
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    
    n_train = len(train_df)
    x_train = np.reshape(np.array(train_df.x), (n_train, 1))
    y_train = np.reshape(np.array(train_df.y), (n_train, 1))

    n_test = len(test_df)
    x_test = np.reshape(np.array(test_df.x), (n_test, 1))
    y_test = np.reshape(np.array(test_df.y), (n_test, 1))

    return x_train, y_train, x_test, y_test


def read_classification_data(dataset_name):
    train_path = os.path.join(DATA_PATH, 'classification',
                              f'{dataset_name}-training.csv')
    test_path = os.path.join(DATA_PATH, 'classification',
                             f'{dataset_name}-test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train = len(train_df)
    x_train = np.concatenate((np.expand_dims(train_df.x, axis=-1),
                              np.expand_dims(train_df.y, axis=-1)), axis=1)
    y_train = np.reshape(np.array(train_df.c), (n_train, 1))
    oh = OneHotEncoder()
    y_train = oh.fit_transform(y_train).toarray()

    n_test = len(test_df)
    x_test = np.concatenate((np.expand_dims(test_df.x, axis=-1),
                              np.expand_dims(test_df.y, axis=-1)), axis=1)
    y_test = np.reshape(np.array(test_df.c), (n_test, 1))
    y_test = oh.fit_transform(y_test).toarray()

    return x_train, y_train, x_test, y_test


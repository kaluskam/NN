import numpy as np

from network import NN
from activation_functions import Sigmoid, Linear, Softmax


def test_layers_multiplication():
    pass


def test_errors_calculation(nn, x, y, y_pred):
    errors = nn.calculate_errors(y, y_pred)

    assert np.allclose(errors[2], np.array([0.32320398, -1.0634558]))
    assert np.allclose(errors[1], np.array([0.04098044, -0.0905149, -0.09409743]))
    assert np.allclose(errors[0], np.array([-0.00012218,  0.07091215]))


def test_backpropagation(nn, x, y, y_pred):
    nn.batch_size = 1

    delta = nn.propagate_backwards(x=x, y_true=y, y_pred=y_pred)
    assert np.allclose(delta['weights'][2], np.array([[0.3133519, -1.03103897],
       [0.29284274, -0.96355653],
       [0.26604754, -0.87539081]]))
    assert np.allclose(delta['weights'][1],
                       np.array([[0.04087911, -0.09029109, -0.09386476],
                                 [0.0299591, -0.06617169, -0.06879073]]))
    assert np.allclose(delta['weights'][0],
                       np.array([[-8.93207371e-05,  5.18409356e-02],
                                [-1.07615787e-04,  6.24592145e-02]]))


def test_regression_network():
    x = np.array([[1, 2], [1, 1]])
    y = np.array([[0, 1], [0, -1]])

    nn = NN(x.shape, neurons_num=[2, 3, 2],
            activations=[Sigmoid(), Sigmoid(), Linear()])

    nn.layers[0].weights = np.array([[1, -1], [3, 2]])
    nn.layers[1].weights = np.array([[1, 1, 0], [2, -1, -2]])
    nn.layers[2].weights = np.array([[1, -1], [0, 1], [-2, 0]])

    nn.layers[0].biases = np.array([-1, -2])
    nn.layers[1].biases = np.array([1, 2, 3])
    nn.layers[2].biases = np.array([1, 0])

    test_predict_function(nn, x)
    y_pred = nn.predict(x[0])
    test_errors_calculation(nn, x[0], y[0], y_pred)
    test_backpropagation(nn, x[0], y[0], y_pred)


def test_predict_function(nn, x):

    y_pred = nn.predict(x[0])
    assert np.array_equal(nn.recent_calculations[1], np.array([6, 1]))
    assert np.allclose(nn.recent_calculations[2], np.array(
        [3.45964454, 2.2664688, 1.53788284]), atol=1e-06)
    assert np.allclose(nn.recent_calculations[3], np.array(
        [0.32320398, -0.0634558]))

    assert np.allclose(y_pred, np.array(
        [0.32320398, -0.0634558]))


def test_classification_error_calculation():
    x = np.array([2, -1])
    y = np.array([1, 0])

    nn = NN(input_shape=[1, 2], neurons_num=[3, 2],
            activations=[Sigmoid(), Softmax()])
    nn.loss = 'cross_entropy'
    nn.batch_size = 1

    nn.layers[0].weights = np.array([[1, 1, 0], [2, -1, -2]])
    nn.layers[1].weights = np.array([[1, -1], [0, 1], [-2, 0]])

    nn.layers[0].biases = np.array([-1, -2, 1])
    nn.layers[1].biases = np.array([1, 2])

    y_pred = nn.predict(x)
    errors = nn.calculate_errors(y_true=y, y_pred=y_pred)
    deltas = nn.propagate_backwards_clf(y, y_pred, x)
    assert np.allclose(errors[1],  np.array([-0.95682356,  0.95682356]))
    assert np.allclose(deltas['weights'][1], np.array([[-0.25732949,  0.25732949],
       [-0.69949407,  0.69949407],
       [-0.91144537,  0.91144537]]))


def test_network_creation():
    for function in [Sigmoid(), Softmax(), Linear()]:
        nn = NN(input_shape=[1, 2], neurons_num=[5, 2],
                activations=[Sigmoid(), function])
        assert nn.layers[0].weights.shape == (2, 5)
        assert nn.layers[1].weights.shape == (5, 2)
        assert nn.layers[0].biases.shape == (5, 1)
        assert nn.layers[1].biases.shape == (2, 1)


def test_errors():
    y_true = np.ones((2, 1))
    x = np.ones((2, 1))
    for function in [Sigmoid(), Softmax(), Linear()]:
        print('testing for: ', function.__class__)
        nn = NN(input_shape=[1, 2], neurons_num=[5, 2],
                activations=[Sigmoid(), function])
        nn.predict(x)
        errors = nn.calculate_errors(y_true)
        assert len(errors) == 2

        assert errors[0].shape == (5, 1)
        assert errors[1].shape == (2, 1)


def test_propagate_backwards():
    y_true = np.ones((2, 1))
    x = np.ones((2, 1))
    for function in [Sigmoid(), Softmax(), Linear()]:
        print('testing for: ', function.__class__)
        nn = NN(input_shape=[1, 2], neurons_num=[5, 2],
                activations=[Sigmoid(), function])
        nn.predict(x)
        deltas = nn.propagate_backwards(y_true, x)
        assert len(deltas['weights']) == 2
        assert len(deltas['biases']) == 2

        assert deltas['weights'][0].shape == (2, 5)
        assert deltas['weights'][1].shape == (5, 2)

        assert deltas['biases'][0].shape == (5, 1)
        assert deltas['biases'][1].shape == (2, 1)
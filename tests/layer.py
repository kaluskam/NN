import numpy as np

from layer import Layer
from activation_functions import Sigmoid, Linear, Softmax


def test_layer_calculate_function():
    layer1 = Layer(shape=(2, 5), activation=Sigmoid())
    x = np.ones((2, 1))

    z1 = layer1.calculate(x)
    assert z1.shape == (5, 1)

    layer2 = Layer(shape=(5, 3), activation=Sigmoid())
    z2 = layer2.calculate(z1)
    assert z2.shape == (3, 1)


def test_layer_activate_function():
    for function in [Sigmoid(), Linear(), Softmax()]:
        layer1 = Layer(shape=(2, 5), activation=function)
        x = np.ones((2, 1))

        z1 = layer1.calculate(x)
        a1 = layer1.activate(z1)
        assert a1.shape == (5, 1)

        layer2 = Layer(shape=(5, 3), activation=function)
        z2 = layer2.calculate(a1)
        a2 = layer2.activate(z2)
        assert a2.shape == (3, 1)


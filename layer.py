import numpy as np

from activation_functions import Sigmoid


class Layer:
    def __init__(self, shape, activation=Sigmoid()):
        self.shape = shape
        self.activation = activation
        self._initialize_weights()

    def _initialize_weights(self, min_val=-0.5, max_val=0.5):
        self.weights = np.random.uniform(min_val, max_val, size=self.shape)
        self.biases = np.random.uniform(min_val, max_val, size=self.shape[1])

    def calculate(self, x):
        return np.matmul(x, self.weights) + self.biases

    def activate(self, x):
        return self.activation.calculate(x)

    def update(self, delta_weights, delta_biases):
        self.weights += np.array(np.reshape(delta_weights, self.shape))
        self.biases += np.array(np.reshape(delta_biases, self.shape[1]))
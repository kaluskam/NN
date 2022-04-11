import numpy as np


class ActivationFunction:
    def calculate(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivationFunction):

    def calculate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return np.exp(-x) / np.square((1 + np.exp(-x)))


class Linear(ActivationFunction):

    def calculate(self, x):
        return x

    def derivative(self, x):
        return np.ones(shape=(x.shape[-1], 1))


class Softmax(ActivationFunction):

    def calculate(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def derivative(self, x):
        pass
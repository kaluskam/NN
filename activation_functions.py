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
        if x.shape[1] == 1:
            return np.exp(x) / np.sum(np.exp(x))
        if x.shape[1] > 1:
            return np.exp(x) / (np.sum(np.exp(x), axis=1).reshape(
                (x.shape[0], 1)) @ np.ones((1, x.shape[1])))

    def derivative(self, x):
        return 1


class Tanh(ActivationFunction):

    def calculate(self, x):
        counter = np.exp(x) - np.exp(-x)
        denominator = np.exp(x) + np.exp(-x)
        return counter / denominator

    def derivative(self, x):
        return 1 - np.square(self.calculate(x))


class ReLU(ActivationFunction):

    def calculate(self, x):
        return np.where(x > 0, x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

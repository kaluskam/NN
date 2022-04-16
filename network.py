import numpy as np

from layer import Layer
from activation_functions import Softmax
from metrics import mse, cross_entropy_derivative, cross_entropy


class NN:
    def __init__(self, input_shape, neurons_num, activations, seed=123):
        self.batches = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.delta_weights = None
        self.input_shape = input_shape
        self.layers_num = len(neurons_num)
        self.neurons_num = neurons_num
        self.activations = activations
        np.random.seed(seed)
        self._build()

    def _build(self):
        self.layers = []

        layer = Layer(shape=(self.input_shape[1], self.neurons_num[0]),
                      activation=self.activations[0])
        self.layers.append(layer)

        for i in range(1, self.layers_num):
            layer = Layer(
                shape=(self.layers[i - 1].shape[1], self.neurons_num[i]),
                activation=self.activations[i])
            self.layers.append(layer)

    # def feed_forward(self, x):
    #     self.recent_calculations = []
    #     self.recent_calculations.append(x)
    #
    #     x = self.layers[0].calculate(x)
    #     self.recent_calculations.append(x)
    #     x = self.layers[0].activate(x)
    #
    #     for i in range(1, self.layers_num):
    #         x = self.layers[i].calculate(x)
    #         self.recent_calculations.append(x)
    #         x = self.layers[i].activate(x)
    #
    #     return x

    def calculate_errors(self, y_true):
        # y_true = np.expand_dims(y_true, axis=1)
        # y_pred = np.expand_dims(y_pred, axis=1)
        errors = []
        last_error = self.layers[-1].output - y_true
        last_derivative = self.layers[-1].activation.derivative(
            self.layers[-1].weighted_input
        )
        errors.append(np.multiply(last_error, last_derivative))
        for i in range(self.layers_num - 2, -1, -1):
            layer = self.layers[i]
            derivative = layer.activation.derivative(
                layer.weighted_input
            )
            errors.append(np.multiply(
                derivative, (self.layers[i + 1].weights @ errors[-1])))

        errors.reverse()
        return errors

    # def calculate_errors(self, y_true, y_pred):
    #     errors = []
    #     for i in range(self.layers_num - 1, -1, -1):
    #         layer = self.layers[i]
    #         if isinstance(layer.activation, Softmax):
    #             pass
    #             # derivative = np.reshape(
    #             #     cross_entropy_derivative(y_true, y_pred), (1, 1))
    #         else:
    #             derivative = layer.activation.derivative(
    #                 self.recent_calculations[i + 1].reshape(layer.shape[-1],
    #                                                         1))
    #
    #         if i == self.layers_num - 1:
    #             if self.loss.__name__ == 'mse':
    #                 errors.append(np.multiply((y_pred - y_true), derivative)
    #                               .reshape(1, y_true.shape[-1]).squeeze())
    #             if self.loss.__name__ == 'cross_entropy':
    #                 softmax_output = Softmax().calculate(self.recent_calculations[-1])
    #                 errors.append((softmax_output - y_true)
    #                               .reshape(1, y_true.shape[-1]).squeeze())
    #         else:
    #             errors.append(np.multiply(derivative.squeeze(),
    #                                       np.dot(self.layers[i + 1].weights,
    #                                              errors[
    #                                                  -1]).squeeze()).squeeze())
    #     errors.reverse()
    #     return errors

    def propagate_backwards(self, y_true, x):
        delta = {'weights': [], 'biases': []}

        errors = self.calculate_errors(y_true)
        # delta['weights'].insert(0, -self.layers[-2].output @ errors[-1].T)
        # delta['biases'].insert(0, -errors[-1])

        for i in range(self.layers_num - 1, 0, -1):
            delta['weights'].insert(0,
                                    -self.layers[i - 1].output @ errors[i].T)
            delta['biases'].insert(0, -errors[i])

        delta['weights'].insert(0,
                                -x @ errors[0].T)
        delta['biases'].insert(0, -errors[0])
        return delta

    # def propagate_backwards(self, y_true, y_pred, x):
    #     delta = {'weights': [], 'biases': []}
    #
    #     errors = self.calculate_errors(y_true, y_pred)
    #     for i in range(self.layers_num - 1, 0, -1):
    #         a = self.layers[i - 1].activate(
    #             self.recent_calculations[i].reshape(
    #                 self.layers[i - 1].shape[-1], 1))
    #         delta['weights'].insert(0,
    #                                 np.outer(a, errors[i]) / self.batch_size)
    #         # delta['weights'].insert(0,
    #         #                         errors[i].dot(a) / self.batch_size)
    #         delta['biases'].insert(0, (errors[i] / self.batch_size).reshape(-1, 1))
    #
    #     a = self.layers[0].activate(
    #         self.recent_calculations[0].reshape(
    #             x.shape[-1], 1))
    #     delta['weights'].insert(0, np.outer(a, errors[0]) / self.batch_size)
    #     delta['biases'].insert(0, (errors[0] / self.batch_size).reshape(-1, 1))
    #
    #     return delta

    def convert_to_numpy_array(self, x_train, y_train, x_test, y_test):
        if x_test is None or y_test is None:
            return np.reshape(np.array(x_train), (-1, self.input_shape[1])), \
                   np.reshape(np.array(y_train), (-1, y_train.shape[1])), None, None
        else:
            return np.reshape(np.array(x_train), (-1, self.input_shape[1])),\
                   np.reshape(np.array(y_train), (-1, y_train.shape[1])), \
                   np.reshape(np.array(x_test), (-1, self.input_shape[1])), \
                   np.reshape(np.array(y_test), (-1, y_test.shape[1]))

    def update_layers(self):
        for i in range(self.layers_num):
            self.layers[i].update(self.delta_weights['weights'][i],
                                  self.delta_weights['biases'][i])

    def initialize_dict(self):
        d = {'weights': [], 'biases': []}
        for layer in self.layers:
            d['weights'].append(np.zeros(shape=layer.weights.shape))
            d['biases'].append(np.zeros(shape=layer.biases.shape))

        return d

    def sum_dicts(self, dict1, dict2, dict1_multiplier=1, dict2_multiplier=1):

        d_sum = self.initialize_dict()
        for i in range(self.layers_num):
            w_shape = dict1['weights'][i].shape
            d_sum['weights'][i] = dict1_multiplier * dict1['weights'][
                i] + dict2_multiplier * dict2['weights'][i].reshape(w_shape)
            d_sum['biases'][i] = dict1_multiplier * dict1['biases'][
                i] + dict2_multiplier * dict2['biases'][i]
        return d_sum

    def print_results(self, epoch):
        print(f'Epoch number {epoch}/{self.n_epochs}')
        metric_name = self.metric.__name__
        print(f'Loss on training set: '
              f'{self.loss(self.y_train, self.predict(self.x_train))}',
              end=' ')
        print(f'{metric_name} on training set: '
              f'{self.metric(self.y_train, self.predict(self.x_train))}', end='')
        if self.x_test is not None:
            print(f', loss on test set: '
                  f'{self.loss(self.y_test, self.predict(self.x_test))}',
                  end=' ')
            print(f'{metric_name} on test set: '
                  f'{self.metric(self.y_test, self.predict(self.x_test))}')

    def generate_batches(self):
        np.random.shuffle(self.indices)
        indices_permutation = np.split(self.indices,
                                       [i * self.batch_size for i in
                                       range(1, self.n // self.batch_size)])
        return indices_permutation
        # self.batches = {'x': self.x_train[indices_permutation],
        #                 'y': self.y_train[indices_permutation]}

    def fit(self, x_train, y_train, batch_size, n_epochs, learning_rate=0.003,
            x_test=None, y_test=None, loss=None, metric=None, verbose_step=10):

        self.x_train, self.y_train, self.x_test, self.y_test = self.convert_to_numpy_array(
            x_train, y_train, x_test, y_test)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n = self.y_train.shape[0]
        self.indices = np.arange(self.n)
        self.loss = loss
        self.metric = metric

        epoch = 1
        while epoch <= self.n_epochs:
            batches = self.generate_batches()

            #for x_batch, y_batch in zip(self.batches['x'], self.batches['y']):
            for batch in batches:
                self.delta_weights = self.initialize_dict()
                for j in range(self.batch_size):
                    idx = batch[j]
                    x = x_train[idx]
                    y = y_train[idx]
                    self.propagate_forward(np.reshape(x, (-1, 1)))

                    delta = self.propagate_backwards(
                        y_true=np.reshape(y, (-1, 1)),
                        x=np.reshape(x, (-1, 1)))

                    self.delta_weights = self.sum_dicts(
                        dict1=self.delta_weights, dict2=delta,
                        dict2_multiplier=self.learning_rate)

                self.update_layers()

            if epoch % verbose_step == 0:
                self.print_results(epoch)
            epoch += 1

    def propagate_forward(self, x):
        for i in range(0, self.layers_num):
            x = self.layers[i].calculate(x)
            x = self.layers[i].activate(x)

        return x

    def predict(self, x):
        n = x.shape[0]
        for i in range(0, self.layers_num):
            x = x @ self.layers[i].weights + np.ones(shape=(n, 1)) @ self.layers[i].biases.reshape((1, -1))
            x = self.layers[i].activation.calculate(x)
        return x

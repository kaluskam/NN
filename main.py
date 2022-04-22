from network import NN
from activation_functions import Sigmoid, Linear, Softmax, ReLU, Tanh
from prepare_data import read_regression_data, read_classification_data
from metrics import mse, cross_entropy, f_score

# x_train, y_train, x_test, y_test = read_classification_data(
#     dataset_name='easy')

x_train, y_train, x_test, y_test = read_regression_data(
    dataset_name='multimodal-large', index_col=None)

architecture1 = [32, 1]
architecture2 = [40, 40, 1]
architecture3 = [32, 32, 16, 1]

nn = NN(input_shape=x_train.shape, neurons_num=architecture1,
        activations=[ReLU(), Linear()])

nn.fit(x_train, y_train, batch_size=4, n_epochs=80, learning_rate=0.0001, loss=mse, metric=mse, x_test=x_test, y_test=y_test, verbose_step=10)
#nn.fit(x_train, y_train, batch_size=4, n_epochs=100, learning_rate=0.00005, loss=mse, metric=mse, x_test=x_test, y_test=y_test, verbose_step=10)
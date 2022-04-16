from network import NN
from activation_functions import Sigmoid, Linear, Softmax, ReLU, Tanh
from prepare_data import read_regression_data, read_classification_data
from metrics import mse, cross_entropy, f_score

x_train, y_train, x_test, y_test = read_classification_data(
    dataset_name='easy')

# x_train, y_train, x_test, y_test = read_regression_data(
#     dataset_name='square-simple')

nn = NN(input_shape=x_train.shape, neurons_num=[32, 32, 2],
        activations=[ReLU(), ReLU(), Softmax()], seed=42)

nn.fit(x_train, y_train, batch_size=8, n_epochs=40,
       learning_rate=0.0001,
       x_test=x_test, y_test=y_test, loss=cross_entropy, metric=f_score)

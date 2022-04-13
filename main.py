from network import NN
from activation_functions import Sigmoid, Linear, Softmax
from prepare_data import read_regression_data, read_classification_data
from metrics import mse, cross_entropy, f1_score

x_train, y_train, x_test, y_test = read_classification_data(
    dataset_name='easy')
#
# x_train, y_train, x_test, y_test = read_regression_data(
#     dataset_name='square-simple')

nn = NN(input_shape=x_train.shape, neurons_num=[32, 2],
        activations=[Sigmoid(), Softmax()], seed=42)

nn.fit(x_train, y_train, batch_size=y_train.shape[0], n_epochs=20,
       learning_rate=0.001,
       x_test=x_test, y_test=y_test, loss=cross_entropy, metric=f1_score)

from network import NN
from activation_functions import Sigmoid, Linear, Softmax
from loss_funtions import mse, cross_entropy
from prepare_data import read_regression_data, read_classification_data

x_train, y_train, x_test, y_test = read_classification_data(
    dataset_name='easy')

nn = NN(input_shape=x_train.shape, neurons_num=[64, 32, 16, 2],
        activations=[Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()])

nn.fit(x_train, y_train, batch_size=y_train.shape[0], n_epochs=1000,
       learning_rate=0.0003,
       x_test=x_test, y_test=y_test, loss=mse)

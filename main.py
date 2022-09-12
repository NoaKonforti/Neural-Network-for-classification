"""
  imports
  """
import numpy as np
import tests
import scipy.io
from layers import Dense, Residual
from activations import Tanh, Softmax, ReLU, Sigmoid
from loss import SoftMaxRegression
from network import Network
from tests import loss_grad_test_b, loss_grad_test_w, SGD


"""
  load the data
     """
Peaks = scipy.io.loadmat('PeaksData.mat')
x_train = Peaks.get('Yt')
y_train = np.transpose(Peaks.get('Ct'))
x_test = Peaks.get('Yv')
y_test = np.transpose(Peaks.get('Cv'))

"""
  parameters
  """
n = 4000
epochs = 300
lr = 0.001
batch_size = 50

if __name__ == '__main__':

    """
      build the model
      """
    layers = [(Dense(units=6), Tanh()),
             (Residual(units=8, activation_func=ReLU()), ReLU()),
              (Dense(units=4), Tanh()),
              (Dense(units=12), Tanh()),
              (Dense(units=3), Tanh()),
              (Dense(units=7), Tanh()),
              (Dense(units=16), None)]
    loss = SoftMaxRegression(n_classes=5)
    model = Network(layers=layers, loss=loss)
    """
      train the model
      """
    model.train(x_train, y_train, epochs=epochs, learning_rate=lr, batch_size=batch_size)


    """
      test the model
    """
    test_score = model.predict(x_test, y_test)
    print( "test score = ", test_score)
    predict = tests.SGD_network(model, x_test, y_test)
    print("predict = ", predict)


    """""
    Part 1 - the classifier and optimizer
    """
# Create the data for small least squares example:
     X1 = np.random.normal(0, 1, (4, n))
     X2 = np.random.normal(10, 5, (4, n))
     Y1 = np.zeros(n)
     Y2 = np.ones(n)
     X = np.concatenate((X1, X2), axis=1)
     Y = np.concatenate((Y1, Y2))
     inds = np.arange(2 * n)
     np.random.shuffle(inds)
     X = X[:, inds]
     Y = Y[inds].astype(int)
     Y_hot = np.zeros((2 * n, 2))
     Y_hot[np.arange(2 * n), Y] = 1

# test the classifier and optimizer
    # myfun = SoftMaxRegression(n_classes=5)
    # myfun(X, Y_hot)

    # tests.loss_grad_test_w(myfun, myfun.gradient, X, Y_hot, myfun.weights, 10)

    # tests.loss_grad_test_b(myfun, myfun.gradient_b, X, Y_hot, myfun.bias, 10)

    # tests.SGD(myfun, myfun.gradient, X, Y_hot, lr, 2, 10)

    # randomize = index = np.random.choice(x_train.shape[1], 50, replace=False)
    # tests.SGD(myfun, myfun.gradient, x_train[:, randomize], y_train[randomize], lr, batch_size, epochs)
    # randomize = index = np.random.choice(x_test.shape[1], 50, replace=False)
    # tests.SGD(myfun, myfun.gradient, x_test[:, randomize], y_test[randomize], lr, batch_size, epochs)

    """""
    Part 2 - the neural network 
    """

     l_dense = Dense(units=3)
    # tests.jacobian_test(l_dense, Tanh(), X)

     l_res = Residual(units=3, activation_func=Tanh())
    # tests.jacobian_test_residual(l_res, Tanh(), X)

    # model.jacobian_test(model.H(), model.dH(), 2)

    # tests.network_grad_test_w(model, x_train, y_train, 30)

     randomize = index = np.random.choice(x_train.shape[1], 200, replace=False)
     x_random_train = x_train[:, randomize]
     y_random_train = y_train[randomize]
     model.train(x_random_train, y_random_train, epochs=epochs, learning_rate=lr, batch_size=batch_size)
     test_score = model.predict(x_test, y_test)
     print( "test score = ", test_score)
    # predict = tests.SGD_network(model, x_test, y_test)
    # print("predict = ", predict)

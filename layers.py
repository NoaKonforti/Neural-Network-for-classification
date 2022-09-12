import numpy as np


class Dense:
    def __init__(self, units: int):
        self.units = units  # Number of neurons
        self.input_units = None
        self.weights = None  # input weights
        self.bias = None
        self.grad_weights = 0
        self.grad_bias = 0
        self.output = None

    def build(self):
        self.weights = np.random.randn(self.input_units, self.units)  # matrix (l-1*l)
        self.bias = np.zeros((self.units,))

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.weights is None:
            _input_shape = input_tensor.shape
            self.input_units = _input_shape[0]
            self.build()

        self.output = (np.dot(input_tensor.transpose(), self.weights) + self.bias).transpose()
        return self.output

    @staticmethod
    def gradient(input, dz, n_examples):
        return (np.dot(dz, np.transpose(input)) / n_examples).transpose()

    def update(self, lr: float):
        self.weights = self.weights - lr * self.grad_weights
        self.bias = self.bias - lr * self.grad_bias


class Residual:
    def __init__(self, units: int, activation_func=None):
        self.units = units  # Number of neurons
        self.input_units = None
        self.weights1 = None
        self.weights2 = None
        self.bias = None
        self.activation_func = activation_func
        self.grad_weights1 = 0
        self.grad_weights2 = 0
        self.grad_bias = 0
        self.grad_input = 0
        self.output = None

    def build(self):
        self.weights1 = np.random.randn(self.input_units, self.units)
        # for same dimension of x as f
        self.weights2 = np.random.randn(self.units, self.input_units)
        self.bias = np.zeros((self.units,))

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        _input_shape = input_tensor.shape

        if self.weights1 is None and self.weights2 is None:
            _input_shape = input_tensor.shape
            self.input_units = _input_shape[0]
            self.build()

        output = (np.dot(input_tensor.transpose(), self.weights1) + self.bias).transpose()
        self.grad_input = output
        if self.activation_func is not None:
            output = self.activation_func(output)
            self.output = input_tensor + np.dot(self.weights2.transpose(), output)

        return self.output

    def update(self, lr: float):
        self.weights1 = self.weights1 - lr * self.grad_weights1
        self.weights2 = self.weights2 - lr * self.grad_weights2
        self.bias = self.bias - lr * self.grad_bias

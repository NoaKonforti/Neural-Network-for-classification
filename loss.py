import numpy as np


class SoftMaxRegression:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.weights = None
        self.grad_weights = None
        self.bias = None
        self.grad_bias = None
        self._input = None

    def build_w(self, input_units):
        self.weights = np.random.randn(input_units, self.n_classes)  # (n*labels)

    def build_b(self):
        self.bias = np.random.randn(self.n_classes)

    def __call__(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self._input = x
        if self.weights is None:
            input_units = x.shape[0]
            self.build_w(input_units)

        if self.bias is None:
            self.build_b()

        # SoftMax Regression function
        x = (np.dot(np.transpose(x), self.weights) + self.bias).transpose()  # (labels*m)
        x = self.softmax(x)
        loss = np.sum(labels.transpose() * np.log(x)) / labels.shape[0]
        return -1 * loss

    # The gradient of softmax regression w.r.t the weights # add b x.
    def gradient(self, labels: np.ndarray) -> np.ndarray:
        output = self.softmax((np.dot(np.transpose(self._input), self.weights) + self.bias).transpose())
        grad_w = self._input @ (np.subtract(output.transpose(), labels)) / labels.shape[0]
        return grad_w  # size of w

    # The gradient of softmax regression w.r.t the bias
    def gradient_b(self, labels: np.ndarray) -> np.ndarray:
        output = self.softmax((np.dot(np.transpose(self._input), self.weights) + self.bias).transpose())
        grad_b = np.mean(np.subtract(output.transpose(), labels), axis=0)
        return grad_b  # size of b (n)

    # The gradient of softmax regression w.r.t to x
    def gradient_x(self, labels: np.ndarray) -> np.ndarray:
        output = self.softmax((np.dot(np.transpose(self._input), self.weights)).transpose())
        grad_x = self.weights @ (np.subtract(output.transpose(), labels)).transpose() / labels.shape[0]
        return grad_x  # size of x (n*m)

    def update(self, lr: float):
        self.weights = self.weights - lr * self.grad_weights
        self.bias = self.bias - lr * self.grad_bias

    def softmax(self, x):
        """
                x: dot product (x, weights)
                """
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x, axis=0) + 1e-4
        sum_exp = np.repeat(sum_exp[:, np.newaxis], self.n_classes, axis=1).transpose()
        return np.divide(exp_x, sum_exp)  # size of (m*l)


import numpy as np

class Softmax:
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        exp_z = np.exp(input_tensor)
        softmax_vec = (exp_z.transpose() / np.sum(exp_z, axis=1)).transpose()
        return softmax_vec

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        s = self(input_tensor)
        ds = -1 * np.outer(s, s) + np.diag(s.flatten())
        return ds


class Sigmoid:
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return 1. / (1 + np.exp(-1 * input_tensor))

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return self(input_tensor) * (1 - self(input_tensor))


class ReLU:
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.maximum(input_tensor, 0)

    @staticmethod
    def gradient(input_tensor: np.ndarray) -> np.ndarray:
        result = input_tensor.copy()
        result[input_tensor >= 0] = 1
        result[input_tensor < 0] = 0
        return result


class Tanh:
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.tanh(input_tensor)

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return 1 - self(input_tensor) ** 2

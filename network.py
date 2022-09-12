import numpy as np
from matplotlib import pyplot as plt

import layers
import tests


class Network:
    def __init__(self, layers, loss):
        """Fully connected neural network with a full forward and backward pass.
        """
        self.n_layers = len(layers)
        self.layers = layers
        self.loss = loss
        self._input = None
        self._num_examples = None

    def forward_step(self, input_tensor: np.ndarray, labels: np.ndarray):
        """
        :param input_tensor: (num_features, num_examples)
        :param labels: (num_examples, num_classes)
        :return: (num_units_of_final_layer, num_examples)
        """
        if self._num_examples is None:
            self._num_examples = input_tensor.shape[1]

        output = input_tensor

        for layer, activation in self.layers:
            output = layer(output)
            if activation is not None:
                output = activation(output)

        return self.loss(output, labels)

    def backward_step(self, batch: np.ndarray, labels: np.ndarray):
        """
               :param batch: (num_features, num_examples)
               :param output: (num_features_output (last layer before loss layer), num_examples)
               :param labels: (num_examples, num_classes)
               :param batch_size:
               """
        layers_weights_grad = []
        self.loss.grad_weights = self.loss.gradient(labels)
        layers_weights_grad.insert(0, self.loss.grad_weights)
        self.loss.grad_bias = self.loss.gradient_b(labels)
        da = self.loss.gradient_x(labels)

        # going over the layers from last to first
        for i in range(self.n_layers-1, -1, -1):
            layer, activation = self.layers[i]

            if i == 0:
                prev_layer_output = batch
            else:
                prev_layer, prev_activation = self.layers[i-1]
                if prev_activation is not None:
                    prev_layer_output = prev_activation(prev_layer.output)
                else:
                    prev_layer_output = prev_layer.output

            if isinstance(layer, layers.Dense): # Dense layer
                if activation is not None:
                    dz = activation.gradient(layer.output)
                    dz = np.multiply(dz, da)
                else:
                    dz = da

                layer.grad_weights = layer.gradient(prev_layer_output, dz, self._num_examples)
                layers_weights_grad.insert(0, layer.grad_weights)
                layer.grad_bias = np.mean(dz, axis=1)
                da = np.dot(layer.weights, dz)


            else: # Residual layer
                if activation is not None:
                    d = activation.gradient(layer.grad_input)
                   # dz = np.dot(layer.weights2, np.dot(dw1.transpose(), layer.weights1).transpose())
                    dz = np.dot(np.multiply(layer.weights2, layer.weights1.transpose()).transpose(), d)
                    dz = np.multiply(da, dz)
                else:
                    dz = da

                layer.grad_weights1 = np.dot(np.multiply(d, np.dot(layer.weights2, dz)), prev_layer_output.transpose()).transpose() / self._num_examples
                layer.grad_weights2 = np.dot(dz, activation(layer.grad_input).transpose()).transpose() / self._num_examples
                weights_res = np.array([layer.grad_weights1, layer.grad_weights2])
                layers_weights_grad.insert(0, weights_res)
                layer.grad_bias = np.mean(d, axis=1)
                da = np.dot(np.dot( layer.weights1, layer.weights2), dz)


        return np.asarray(layers_weights_grad, dtype=object)

    def train(self, examples: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int, learning_rate: float):
        num_batches = int(np.ceil(examples.shape[1] / batch_size))
        loss_vec = np.zeros(epochs)
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            # randomizing & splitting the batches
            randomize = np.arange(examples.shape[1])
            np.random.shuffle(randomize)
            lb_batch = labels[randomize]
            ex_batch = examples[:, randomize]

            for batch_ind in range(num_batches):
                interval = batch_ind * batch_size
                if batch_ind + 1 == num_batches:
                    current_batch = ex_batch[:, interval:]
                    current_labels = lb_batch[interval:]
                else:
                    current_batch = ex_batch[:, interval:interval+batch_size]
                    current_labels = lb_batch[interval:interval+batch_size]
                # Forward the data in th network
                loss = self.forward_step(current_batch, current_labels)
                epoch_loss += loss
                # Calculate all gradients for all layers
                _ = self.backward_step(current_batch, current_labels)
                # Update all weights and biases (network's parameters)
                self.update(learning_rate)

            print(f"Epoch: {epoch:03d}, Loss: {epoch_loss/num_batches:0.4f}")
            loss_vec[epoch-1] = epoch_loss /num_batches
        x_plt = [i for i in range(1, epochs + 1)]
        plt.figure(3)
        plt.plot(x_plt, loss_vec)
        plt.title("Successful minimizing the network's loss plot")
        plt.xlabel("epoch number")
        plt.ylabel("loss")
        plt.show()

    def predict(self, examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        outputs = self.forward_step(examples, labels)
        return (outputs > 0.5).astype("uint8")


    def update(self, learning_rate):
        self.loss.update(learning_rate)
        for layer, _ in self.layers:
            layer.update(learning_rate)


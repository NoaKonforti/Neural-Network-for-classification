import matplotlib.pyplot as plt
import numpy as np
import layers
import loss

""""
 gradient tests
"""
# gradient test w.r.t. weights
def loss_grad_test_w(f, df, X: np.ndarray, Y_hot: np.ndarray, param, itr: int):
    # params:
    epsilon = 0.01
    f0 = f(X, Y_hot)
    d = np.random.rand(param.shape[0], param.shape[1])
    y0 = np.zeros(itr)
    y1 = np.zeros(itr)
    print("k\terror order 1 \t\t error order 2")
    save = param

    for i in range(0, itr):
        epsilon = epsilon * 0.5
        f.weights = f.weights + epsilon * d
        fk = f(X, Y_hot)
        g1 = df(Y_hot)
        f1 = f0 + epsilon * np.dot((np.hstack(g1).reshape(1, -1)), np.transpose(np.hstack(d)).reshape(-1, 1))
        y0[i] = np.abs(fk - f0)
        y1[i] = np.abs(fk - f1)
        f.weights = save
        print(i + 1, "\t", y0[i], "\t", y1[i])

    x_plt = [i for i in range(1, itr + 1)]
    plt.figure(1)
    plt.semilogy(x_plt, y0)
    plt.semilogy(x_plt, y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Successful Grad test w.r.t weights in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


# gradient test w.r.t. bias
def loss_grad_test_b(f, df, X: np.ndarray, Y_hot: np.ndarray,  param, itr: int):
    # params:
    epsilon = 0.01
    f0 = f(X, Y_hot)
    d = np.random.rand(param.shape[0])
    y0 = np.zeros(itr)
    y1 = np.zeros(itr)
    print("k\terror order 1 \t\t error order 2")
    save = param

    for i in range(0, itr):
        epsilon = epsilon * 0.5
        f.bias = f.bias + epsilon * d
        fk = f(X, Y_hot)
        g1 = df(Y_hot)
        f1 = f0 + epsilon * np.dot((np.hstack(g1).reshape(1, -1)), np.transpose(np.hstack(d)).reshape(-1, 1))
        y0[i] = np.abs(fk - f0)
        y1[i] = np.abs(fk - f1)
        f.bias = save
        print(i + 1, "\t", y0[i], "\t", y1[i])

    x_plt = [i for i in range(1, itr + 1)]
    plt.figure(2)
    plt.semilogy(x_plt, y0)
    plt.semilogy(x_plt, y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Successful Grad test w.r.t bias in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


# gradient test to Dense layer w.r.t. weights
def layer_grad_test(layer, f, df, X: np.ndarray, param, itr: int):
    # params:
    epsilon = 0.01
    f0 = f(layer(X))
    d = np.random.rand(param.shape[0], param.shape[1])
    y0 = np.zeros(itr)
    y1 = np.zeros(itr)
    print("k\terror order 1 \t\t error order 2")
    save = param

    for i in range(0, itr):
        epsilon = epsilon * 0.5
        layer.weights = layer.weights + epsilon * d
        x_stat = layer(X)
        fk = f(x_stat)
        g1 = df(x_stat)
        f1 = f0 + epsilon * np.dot((np.hstack(g1).reshape(1, -1)), np.transpose(np.hstack(d)).reshape(-1, 1))
        y0[i] = np.abs(fk - f0)
        y1[i] = np.abs(fk - f1)
        layer.weights = save
        print(i + 1, "\t", y0[i], "\t", y1[i])

    x_plt = [i for i in range(1, itr + 1)]
    plt.figure(2)
    plt.semilogy(x_plt, y0)
    plt.semilogy(x_plt, y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Successful Grad test on layer in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


# gradient test for residual layer w.r.t. weights
def residual_grad_test(layer, f, df, X: np.ndarray, itr: int):
    # params:
    epsilon = 0.01
    f0 = f(layer(X))
    d1 = np.random.rand(layer.weights1.shape[0], layer.weights1.shape[1])
    d2 = np.random.rand(layer.weights2.shape[0], layer.weights2.shape[1])
    y0 = np.zeros(itr)
    y1 = np.zeros(itr)
    print("k\terror order 1 \t\t error order 2")
    save1 = layer.weights1
    save2 = layer.weights2

    for i in range(0, itr):
        epsilon = epsilon * 0.5
        layer.weights1 = layer.weights1 + epsilon * d1
        layer.weights2 = layer.weights2 + epsilon * d2
        x_tent = layer(X)
        fk = f(x_tent)
        g1, g2 = df(x_tent)
        f1 = f0 + epsilon * (np.dot(np.concatenate((np.hstack(g1), np.hstack(g2.transpose())), axis=0).reshape(1, -1), np.concatenate((np.hstack(d1), np.hstack(d2.transpose())), axis=0).reshape(-1, 1)))
        #f1 = f0 + epsilon * np.dot(np.hstack(np.dot(g1, g2)).reshape(1, -1), np.hstack(np.dot(d1, d2)).reshape(-1, 1))
        y0[i] = np.abs(fk - f0)
        y1[i] = np.abs(fk - f1)
        layer.weights1 = save1
        layer.weights2 = save2
        print(i + 1, "\t", y0[i], "\t", y1[i])

    x_plt = [i for i in range(1, itr + 1)]
    plt.figure(2)
    plt.semilogy(x_plt, y0)
    plt.semilogy(x_plt, y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Grad test on layer in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


# gradient test for the entire network w.r.t. weights
def network_grad_test_w(network, X: np.ndarray, Y_hot: np.ndarray, itr: int):
    f = network.forward_step
    df = network.backward_step
    f0 = f(X, Y_hot)

    weights = []
    d = []
    for layer, _ in network.layers:
        if isinstance(layer, layers.Dense):
            weights.append(layer.weights)
            d.append(np.random.rand(layer.weights.shape[0], layer.weights.shape[1]))
        else:
            w_res = np.array([layer.weights1, layer.weights2])
            weights.append(w_res)
            d1 = np.random.rand(layer.weights1.shape[0], layer.weights1.shape[1])
            d2 = np.random.rand(layer.weights2.shape[0], layer.weights2.shape[1])
            d.append(np.array([d1, d2]))
    weights.append(network.loss.weights)
    d.append(np.random.rand(network.loss.weights.shape[0], network.loss.weights.shape[1]))
    weights = np.asarray(weights, dtype=object)
    d = np.asarray(d, dtype=object)

    # params:
    epsilon = 0.01
    y0 = np.zeros(itr)
    y1 = np.zeros(itr)
    print("k\terror order 1 \t\t error order 2")

    for i in range(0, itr):
        epsilon = epsilon * 0.5
        ind = 0

        for layer, _ in network.layers:
            if isinstance(layer, layers.Dense):
                layer.weights = layer.weights + epsilon * d[ind]
            else:
                layer.weights1 = layer.weights1 + epsilon * d[ind]
                layer.weights2 = layer.weights2 + epsilon * d[ind+1]
                ind = ind + 1
            ind = ind + 1
        network.loss.weights = network.loss.weights + epsilon * d[-1]

        fk = f(X, Y_hot)
        g1 = df(X, Y_hot)
        accu = 0
        for ind in range(network.n_layers):
                accu += np.dot((np.hstack(g1[ind]).reshape(1, -1)), np.transpose(np.hstack(d[ind])).reshape(-1, 1))
        ind = 0
        f1 = f0 + epsilon * accu
        y0[i] = np.abs(fk - f0)
        y1[i] = np.abs(fk - f1)
        for layer, _ in network.layers:
            if isinstance(layer, layers.Dense):
                layer.weights = weights[ind]
            else:
                layer.weights1 = weights[ind]
                layer.weights2 = weights[ind + 1]
                ind = ind + 1
            ind = ind + 1
        network.loss.weights = weights[-1]
        print(i + 1, "\t", y0[i], "\t", y1[i])

    x_plt = [i for i in range(1, itr + 1)]
    plt.figure(1)
    plt.semilogy(x_plt, y0)
    plt.semilogy(x_plt, y1)
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Grad test w.r.t weights in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()

""""
 SGD test
"""
# test SGD loss function
def SGD(f, df, x: np.ndarray, labels: np.ndarray, lr, batch_size, ep):
    loss = f(x, labels)
    print("Loss before SGD: %.3f" % (loss))
    num_batches = int(x.shape[1] / batch_size)
    loss_vec = np.zeros(ep)
    succ_vec = np.zeros(ep)
    # loop over epochs
    for i in range(0, ep):
        # divide the data into random mini batches
        randomize = np.arange(x.shape[1])
        np.random.shuffle(randomize)
        lb_batch = labels[randomize]
        ex_batch = x[:, randomize]

        # for mini batches
        for batch_ind in range(0, num_batches):
            interval = batch_ind * batch_size
            if batch_ind + 1 == num_batches:
                current_batch = ex_batch[:, interval:]
                current_labels = lb_batch[interval:]
            else:
                current_batch = ex_batch[:, interval:interval + batch_size]
                current_labels = lb_batch[interval:interval + batch_size]
            f._input = current_batch
            dw = df(current_labels)
            f.weights = f.weights - lr * dw

        loss = f(ex_batch, lb_batch)
        loss_vec[i] = loss
        output = (np.dot(np.transpose(ex_batch), f.weights) + f.bias).transpose()
        output = f.softmax(output)
        predict = (output > 0.5).astype("uint8").transpose()
        predict = np.sum(predict == lb_batch) / lb_batch.shape[1]
        predict = predict / lb_batch.shape[0]
        succ_vec[i] = predict
        print("Epoch: %d, Loss: %.3f" % (i+1, loss))

    x_plt = [i for i in range(1, ep + 1)]
    plt.figure(3)
    plt.plot(x_plt, loss_vec)
    plt.title("Successful minimizing an objective func using SGD plot")
    plt.xlabel("epoch number")
    plt.ylabel("loss")
    plt.show()

    plt.figure(4)
    plt.plot(x_plt, succ_vec, 'o', color='black')
    plt.title("Success percentages of minimizing an objective func using SGD plot")
    plt.xlabel("epoch number")
    plt.ylabel("success percentage")
    plt.show()


def SGD_network(model, input_tensor: np.ndarray, labels: np.ndarray):
    if model._num_examples is None:
        model._num_examples = input_tensor.shape[1]

    output = input_tensor

    for layer, activation in model.layers:
        output = layer(output)
        if activation is not None:
            output = activation(output)

    output = (np.dot(np.transpose(output), model.loss.weights) + model.loss.bias).transpose()
    output = model.loss.softmax(output)
    predict = (output > 0.5).astype("uint8").transpose()
    predict = np.sum(predict == labels) / labels.shape[1]
    predict = predict / labels.shape[0]
    return predict

""""
 jacobian tests
"""
# jacobian test
def jacobian_test(layer, H, x):
    v = layer(x)
    u = np.ones((v.shape[0], v.shape[1]))
    u_vec = u.reshape(-1, 1)

    def G(v):
        return np.dot(np.hstack(H(v)).reshape(1, -1), u_vec)

    def dG(v):
        return ((np.multiply(H.gradient(v), u)) @ x.transpose()).transpose()

    layer_grad_test(layer, G, dG, x, layer.weights, 10)


# jacobian test for res layer
def jacobian_test_residual(layer, H, x):


    def G(v):
        u = np.ones((v.shape[0], v.shape[1]))
        u_vec = u.reshape(-1, 1)
        return np.dot(np.hstack(H(v)).reshape(1, -1), u_vec)

    def dG(v):
        d = H.gradient(layer.grad_input)
        dz = np.dot(np.multiply(layer.weights2, layer.weights1.transpose()).transpose(), d)
        dz = np.multiply(dz, v)
        grad_w1 = (np.dot(np.multiply(d, np.dot(layer.weights2, dz)), x.transpose())).transpose() / v.shape[1]
        grad_w2 = np.dot(H(layer.grad_input), dz.transpose()) / v.shape[1]
        return grad_w1, grad_w2

    residual_grad_test(layer, G, dG, x, 10)




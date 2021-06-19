import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


def data_generate():
    num_examples = 100
    X = np.random.randn(num_examples) * 2 - 1
    y = X * X
    X = X[:, np.newaxis]
    y = y[:, np.newaxis]
    return X, y


def data_iter(batch_size, data, label, random_shuffle=False):
    data_len = len(data)
    if random_shuffle:
        random_indices = np.random.permutation(data_len)
        data = data[random_indices]
        label = label[random_indices]
    for i in range(0, data_len, batch_size):
        s = slice(i, i + batch_size)
        yield data[s], label[s]


class sigmoid:
    def forward(self, z):
        # return 1.0 / (1.0 + np.exp(-z))

        return np.exp(z) / (1.0 + np.exp(z))

    def backward(self, z):
        # return
        sg = self.forward(z)
        return sg * (1 - sg)


class tanh:
    def forward(self, z):
        return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))

    def backward(self, z):
        return 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) ** 2


class relu:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        m = np.zeros_like(z)
        m[z >= 0] = 1
        return m


def L1_loss(y_true, y_pre):
    return np.mean(np.abs(y_true - y_pre))


def L2_loss(y_true, y_pre):
    return 1 / 2 * np.sum(np.square(y_true - y_pre))


class MLPRegressor(object):
    def __init__(self, activation, net_layers=[1, 2, 10, 2, 1], classes=10):
        self.net_layers = net_layers
        self.activation = activation()
        self.weights = []
        self.bias = []
        self._initialize_weights_bias()

    def _initialize_weights_bias(self):
        for i in range(len(self.net_layers) - 1):
            xavier = np.sqrt(2/(self.net_layers[i] + self.net_layers[i + 1]))
            self.weights.append(np.random.normal(0,xavier,(self.net_layers[i] , self.net_layers[i + 1])))
            self.bias.append(np.zeros((1, self.net_layers[i + 1])))

    def forward_propagation(self, X):
        a = [X]
        z = []
        for i in range(len(self.weights) - 1):
            z.append(a[-1] @ self.weights[i] + self.bias[i])
            a.append(self.activation.forward(z[-1]))
        z.append(a[-1] @ self.weights[-1] + self.bias[-1])

        return z, a

    def backpropagation(self, z, a, y_true):
        dw = []
        db = []
        delta = [z[-1] - y_true]
        z = z.copy()
        a = a.copy()
        z = z[:-1]
        for z_item, a_item, wieght in zip(z[::-1], a[::-1], self.weights[::-1]):
            dw.append(a_item.T @ delta[-1])
            db.append(np.sum(delta[-1], axis=0, keepdims=True))
            delta.append(delta[-1] @ wieght.T * self.activation.backward(z_item))
        dw.append(a[0].T @ delta[-1])
        db.append(np.sum(delta[-1], axis=0, keepdims=True))
        return dw[::-1], db[::-1]

    def fit(self, features, labels, num_epochs, lr, alpha=0.0001):
        self.loss = []
        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size=1000, data=features, label=labels):
                sample_num = len(X)
                z, a = self.forward_propagation(X)
                dw, db = self.backpropagation(z, a, y)
                assert len(dw) == len(self.weights) and len(db) == len(self.bias)
                self.weights = [w - lr * (dweight / sample_num + alpha * w) for w, dweight in zip(self.weights, dw)]
                self.bias = [bias - lr * dbias / sample_num for bias, dbias in zip(self.bias, db)]
                loss = L2_loss(y_true=y, y_pre=z[-1])
                self.loss.append(loss)
            if epoch % 100 == 0:
                print('epoch %d, loss %f' % (epoch + 1, loss))
        return self

    def predict(self, X):
        z, a = self.forward_propagation(X)
        return z[-1]


if __name__ == '__main__':
    np.random.seed(100)
    X, y = data_generate()
    nn = MLPRegressor(activation=sigmoid, net_layers=[1, 10, 10, 1])
    nn.fit(X, y, num_epochs=1000, lr=0.1)
    f = nn.predict(X)

    plt.title('training_graph')
    y = np.squeeze(y)
    X = np.squeeze(X)
    f = np.squeeze(f)
    indices = np.argsort(X)
    X = X[indices]
    y = y[indices]
    f = f[indices]
    plt.plot(X, y, color='green')
    plt.plot(X, f, color='blue')
    plt.show()

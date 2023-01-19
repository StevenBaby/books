# coding=utf-8

import numpy as np


class Optimizer(object):

    def __init__(self, lr=0.1) -> None:
        self.lr = lr

    def update(self, param: np.ndarray, grad: np.ndarray):
        param -= self.lr * grad


class SGDOptimizer(Optimizer):

    pass


class OptimParams(object):

    optim_class = SGDOptimizer
    params = {
        'lr': 0.1
    }

    @classmethod
    def make_optimizer(cls):
        return cls.optim_class(**cls.params)


class Layer(object):

    def forward(self, x: np.ndarray):
        pass

    def backward(self, dy: np.ndarray):
        pass

    def update(self):
        pass


class LinearLayer(Layer):
    def __init__(self, input, output, params=OptimParams) -> None:
        self.w = np.random.normal(
            loc=0.0,
            scale=pow(input, -0.5),
            size=(input, output)
        )

        self.dw = self.w * 0.1
        self.batch_size = 1
        self.optim = params.make_optimizer()

    def forward(self, x: np.ndarray):
        self.batch_size = x.shape[0]
        if x.ndim == 1:
            self.xshape = x.shape
            self.batch_size = 1
            x = x.reshape(1, -1)

        self.x = x
        y = np.dot(x, self.w)
        return y

    def backward(self, dy):
        dx = np.dot(dy, self.w.T)
        self.dw = np.dot(self.x.T, dy)

        if self.batch_size == 1:
            dx = dx.reshape(self.xshape)
        return dx

    def update(self):
        self.optim.update(self.w, self.dw)


class AffineLayer(LinearLayer):

    def __init__(self, input, output, params=OptimParams) -> None:
        super().__init__(input, output, params)
        self.b = np.random.normal(
            loc=0.0,
            scale=pow(input, -0.5),
            size=output
        )
        self.optimb = params.make_optimizer()

    def forward(self, x: np.ndarray):
        y = super().forward(x)
        y += self.b
        return y

    def backward(self, dy):
        dx = super().backward(dy)
        self.db = np.sum(dy, axis=0)
        return dx

    def update(self):
        super().update()
        self.optimb.update(self.b, self.db)


class SigmoidLayer(Layer):

    def forward(self, x):
        # scipy.special.expit(x)
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx


class MeanSquaredError(Layer):
    def forward(self, x, t):
        self.x = x
        self.t = t
        y = 0.5 * np.sum((x - t) ** 2)
        return y

    def backward(self, dy=1):
        dz = (self.x - self.t) * dy
        return dz


class SoftEntropyLayer(Layer):

    @staticmethod
    def softmax(x: np.ndarray):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    @staticmethod
    def cross_entropy_error(y: np.ndarray, t: np.ndarray):
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        loss = self.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dy=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size * dy
        return dx


class BaseNeuralNetwork(object):

    def __init__(self, input, hidden, output) -> None:
        self.layers = [
            AffineLayer(input, hidden),
            SigmoidLayer(),
            AffineLayer(hidden, output),
            SigmoidLayer(),
        ]

        self.loss_function = MeanSquaredError()

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, y, t):
        err = self.loss_function.forward(y, t)
        return err

    def dloss(self):
        dloss = self.loss_function.backward(1)
        return dloss

    def backward(self, dy):
        for layer in self.layers[::-1]:
            dy = layer.backward(dy)
        return dy

    def update(self):
        for layer in self.layers[::-1]:
            layer.update()

# coding=utf-8

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray):

    return np.maximum(0, x)


def softmax(x: np.ndarray):

    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 误差函数

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
    if t.ndim == 1:
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(
        np.log(
            y[np.arange(batch_size), t] + 1e-7
        )
    ) / batch_size

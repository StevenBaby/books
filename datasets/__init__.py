# coding=utf-8

import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from collections import defaultdict

dirname = os.path.dirname(__file__)
dataset_path = os.path.join(dirname, 'files/minst.pickle')


def oneshot(label):
    one = np.zeros(10) + 0.01
    one[label] = 0.99
    return one


def load_mnist():
    if not os.path.exists(dataset_path):
        print("load mnist datasets from openml....")
        origin = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))

        image = origin.data
        image /= 255.0
        image *= 0.99
        image += 0.01

        label = origin.target.astype(int)
        label = np.array([oneshot(var) for var in label])

        mnist = [
            image,
            label,
        ]
        with open(dataset_path, 'wb') as file:
            file.write(pickle.dumps(mnist))
    else:
        with open(dataset_path, 'rb') as f:
            mnist = pickle.load(f)

    return train_test_split(mnist[0], mnist[1], test_size=0.25)


if __name__ == '__main__':
    load_mnist()

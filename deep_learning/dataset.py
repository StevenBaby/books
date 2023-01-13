# coding=utf-8

import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from collections import defaultdict


dataset_path = 'datasets/minst.pickle'


def load_mnist():
    if not os.path.exists(dataset_path):
        print("load mnist datasets from openml....")
        origin = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))
        mnist = [
            origin.data,
            origin.target.astype(int),
        ]
        with open(dataset_path, 'wb') as file:
            file.write(pickle.dumps(mnist))
    else:
        with open(dataset_path, 'rb') as f:
            mnist = pickle.load(f)

    return train_test_split(mnist[0], mnist[1], test_size=0.25)

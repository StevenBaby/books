# coding=utf-8

import os
from sklearn import datasets
import pickle


def load_mnist():
    dataset_path = 'datasets/minst.pickle'
    if not os.path.exists(dataset_path):
        mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))
        with open(dataset_path, 'wb') as file:
            file.write(pickle.dumps(mnist))
    else:
        with open(dataset_path, 'rb') as f:
            mnist = pickle.load(f)

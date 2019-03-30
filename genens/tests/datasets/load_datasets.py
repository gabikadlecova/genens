# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name):
    'clf:' 'wine' 'breast_cancer' 'digits' 'iris'
    'regr:' 'diabetes' 'boston' 'california (fetch)'

    if dataset_name == 'iris':
        data, target = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=0.33)

    return X_train, y_train, X_test, y_test
# -*- coding: utf-8 -*-

import openml
import os
import pandas as pd
from functools import partial

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder


def load_mnist(split_validation=False, random_state=None, test_size=None):
    features, target = fetch_openml('mnist_784', return_X_y=True)

    if split_validation:
        train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
        return train_X, train_y, test_X, test_y

    features, target = shuffle(features, target, random_state=random_state)
    return features, target


def load_magic(split_validation=False, random_state=None, test_size=None):
    dir_path = os.path.dirname(__file__)

    filename = dir_path + '/magic.csv'

    data = pd.read_csv(filename, sep=',')
    data = shuffle(data)

    features = data[data.columns[:-1]]
    target = data[data.columns[-1]]
    le = LabelEncoder()

    ix = target.index
    target = pd.Series(le.fit_transform(target), index=ix)

    if split_validation:
        train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
        return train_X, train_y, test_X, test_y

    features, target = shuffle(features, target, random_state=random_state)
    return features, target


def load_wilt(split_validation=False, random_state=None, test_size=None):
    use_original_test = split_validation and test_size is None

    dir_path = os.path.dirname(__file__)

    filename = dir_path + '/wilt-train.csv'
    data = pd.read_csv(filename, sep=',')

    test_filename = dir_path + '/wilt-test.csv'
    test_data = pd.read_csv(test_filename, sep=',')

    # concatenate both sets to create the full set
    if not use_original_test:
        data = pd.concat([data, test_data])

    features = data[data.columns[1:]]
    target = data[data.columns[0]]
    le = LabelEncoder()

    ix = target.index
    target = pd.Series(le.fit_transform(target), index=ix)

    # repeat process for validation set, return the original test
    if use_original_test:
        test_X = test_data[test_data.columns[1:]]
        test_y = test_data[test_data.columns[0]]
        le = LabelEncoder()

        ix = test_y.index
        test_y = pd.Series(le.fit_transform(test_y), index=ix)

        return features, target, test_X, test_y

    # choose a different test set
    if split_validation and test_size is not None:
        train_X, test_X, train_y, test_y = train_test_split(features, target,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return train_X, train_y, test_X, test_y

    features, target = shuffle(features, target, random_state=random_state)
    return features, target


def load_from_sklearn(load_func, split_validation=False, random_state=None, test_size=None):
    features, target = load_func(return_X_y=True)

    if split_validation:
        train_X, test_X, train_y, test_y = train_test_split(features, target,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return train_X, train_y, test_X, test_y

    features, target = shuffle(features, target, random_state=random_state)
    return features, target


def load_from_openml(dataset_name, split_validation=False, random_state=None, test_size=None):
    all_datasets = openml.datasets.list_datasets()

    dataset_id = None
    for k, v in all_datasets.items():
        if v['name'] == dataset_name:
            dataset_id = k
            break

    if dataset_id is None:
        raise ValueError("Invalid dataset name.")

    dataset = openml.datasets.get_dataset(dataset_id)

    features, target, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if split_validation:
        train_X, test_X, train_y, test_y = train_test_split(features, target,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return train_X, train_y, test_X, test_y

    features, target = shuffle(features, target, random_state=random_state)
    return features, target


#
load_functions = {}


"""
load_functions = {
    'iris': partial(load_from_sklearn, load_iris),
    'digits': partial(load_from_sklearn, load_digits),
    'wine': partial(load_from_sklearn, load_wine),
    'breast_cancer': partial(load_from_sklearn, load_breast_cancer),
    'wilt': load_wilt,
    'magic': load_magic,
    'mnist': load_mnist
}
"""


def load_dataset(dataset_name, split_validation=False, random_state=None, test_size=None):
    """
    # TODO
    classification: 'wine' 'breast_cancer' 'digits' 'iris' ...
    regression: 'diabetes' 'boston' 'california (fetch)' ...
    """

    if dataset_name in load_functions:
        return load_functions[dataset_name](split_validation=split_validation,
                                            random_state=random_state,
                                            test_size=test_size)
    else:
        return load_from_openml(dataset_name,
                                split_validation=split_validation,
                                random_state=random_state,
                                test_size=test_size)



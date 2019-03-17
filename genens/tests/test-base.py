# -*- coding: utf-8 -*-

import random

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from genens.base import GenensBase
from genens.config import clf_default

if __name__ == "__main__":

    data, target = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33, random_state=42)

    random.seed(420)
    config = clf_default.create_config(data.shape[1])

    bs = GenensBase(config)
    bs.fit(X_train, y_train)

    pass

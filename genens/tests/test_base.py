# -*- coding: utf-8 -*-

import numpy as np
import random

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from genens.base import GenensBase
from genens.config import clf_default
from genens.render import graph

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    data, target = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33)

    config = clf_default.create_clf_config()

    bs = GenensBase(config, pop_size=200, n_jobs=-1)
    bs.set_test_stats(X_train, y_train, X_test, y_test)

    bs.fit(X_train, y_train)

    pred = bs.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(y_test, pred)))

    lbook = bs.logbook
    print(lbook)

    for i, ind in enumerate(bs.pareto):
        graph.create_graph(ind, "best{}.png".format(i))

    gen = lbook.select("gen")
    min_vals = lbook.chapters["score"].select("min")
    avg_vals = lbook.chapters["score"].select("avg")
    max_vals = lbook.chapters["score"].select("max")

    plt.plot(gen, max_vals)
    plt.savefig('fst-graph.png')

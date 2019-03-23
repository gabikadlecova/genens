# -*- coding: utf-8 -*-

import numpy as np
import random

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from genens.base import GenensBase
from genens.config import clf_default
from genens.render import graph

if __name__ == "__main__":
    random.seed(420)
    np.random.seed(42)

    data, target = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33)

    config = clf_default.create_config(data.shape[1])

    bs = GenensBase(config, pop_size=20)

    bs.fit(X_train, y_train)

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

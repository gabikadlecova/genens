# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from genens.base import GenensBase
from genens.config import clf_default
from genens.render import graph

if __name__ == "__main__":
    random.seed(420)
    np.random.seed(420)

    filename = 'wilt-train.csv'
    data = pd.read_csv(filename, sep=',')
    data = shuffle(data, random_state=42)
    X_train = data[data.columns[1:]]
    y_train = data[data.columns[0]]
    le = LabelEncoder()

    ix = y_train.index
    y_train = pd.Series(le.fit_transform(y_train), index=ix)

    test_filename = 'wilt-test.csv'

    data = pd.read_csv(test_filename, sep=',')

    X_test = data[data.columns[1:]]
    y_test = data[data.columns[0]]
    le = LabelEncoder()

    ix = y_test.index
    y_test = pd.Series(le.fit_transform(y_test), index=ix)

    #data, target = load_iris(return_X_y=True)
    #X_train, X_test, y_train, y_test = train_test_split(data, target,
    #                                                    test_size=0.33)

    print('loaded')
    scorer = make_scorer(cohen_kappa_score, weights='quadratic')

    config = clf_default.create_clf_config()

    bs = GenensBase(config, pop_size=200, n_jobs=-1, scorer=scorer)
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

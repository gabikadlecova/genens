# -*- coding: utf-8 -*-

import random
import genens.config.clf_default as defcf
import genens.workflow.model_creation as mc
import genens.render.graph as graph

from genens.gp.operators import gen_half

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn import decomposition
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn import svm
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import tree

from sklearn import neighbors
from sklearn import ensemble


def create_test_config():


    data, target = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33, random_state=42)

    random.seed(42)
    config = defcf.create_config(data.shape[1])

    it = gen_half(20, config, 5, 4)
    for i, res_tree in enumerate(it):
        graph.create_graph(res_tree, "tree{}.png".format(i))
        wf = mc.create_workflow(res_tree, config.func_config)

        try:
            wf.fit(X_train, y_train)
            pred = wf.predict(X_test)
        except Exception as e:
            print(e)
            continue

        print("Score {}: {}".format(i, accuracy_score(y_test, pred)))


if __name__ == "__main__":
    create_test_config()
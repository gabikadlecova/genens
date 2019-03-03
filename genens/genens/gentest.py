# -*- coding: utf-8 -*-

"""Test file for GP tree initialization.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from genens.workflow.builtins import default_config
from genens.workflow.eval import make_workflow

from genens.gp.types import FunctionTemplate, TypeArity, GpTerminal, GpTreeIndividual
from genens.gp.operators import gen_full

import genens.render.graph as graph


def gen_trees():
    
    types = ['out', 'ens', 'data', 'union']
    
    out = []
    
    """bag = FunctionTemplate('bagging', None,
                            [TypeArity('out', (2,'n'))], 'ens')
    
    boost = FunctionTemplate('boosting', None,
                            [TypeArity('out', (2,'n'))], 'ens')"""

    config = default_config()
    config.add_sklearn_ensemble('Ada', AdaBoostClassifier, algorithm='SAMME')
    config.add_model('svc', SVC)
    # config.add_model('knn', KNeighborsClassifier)
    config.add_model('dt', DecisionTreeClassifier)

    boost = FunctionTemplate('Ada', [TypeArity('out', (1,1))], 'ens')
    
    clf1 = GpTerminal('svc', (None, 'ens'))
    # clf2 = GpTerminal('knn', (None, 'ens'))
    clf3 = GpTerminal('dt', (None, 'ens'))

    dummy = GpTerminal('dummy', (None, 'out'))
    
    cEns = FunctionTemplate('_cEns', [TypeArity('ens', (1,1))], 'out')
    
    func_dict = {
            'out' : [ cEns ],
            'ens' : [ boost ]
    }
    
    term_dict = {
            'ens' : [clf1, clf3],
            'out' : [dummy]
    }

    tree = GpTreeIndividual(gen_full(func_dict, term_dict, 6, 4),0)
    wf = make_workflow(tree, config)

    return tree, wf


if __name__ == "__main__":
    for i in range(0, 5):
        res, wf = gen_trees()

        data, target = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size = 0.33, random_state = 42)

        wf.fit(X_train, y_train)
        pred = wf.predict(X_test)

        [print("{}, {}".format(r.name, r.arity)) for r in res.primitives]
        print()
        
        graph.create_graph(res, "tree{}.png".format(i))

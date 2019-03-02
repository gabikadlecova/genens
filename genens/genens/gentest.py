# -*- coding: utf-8 -*-

"""Test file for GP tree initialization.
"""

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
    config.add_sklearn_ensemble('Ada', AdaBoostClassifier)
    config.add_model('svc', SVC)
    config.add_model('knn', KNeighborsClassifier)
    config.add_model('dt', DecisionTreeClassifier)

    boost = FunctionTemplate('Ada', [TypeArity('out', (1,1))], 'ens')
    
    clf1 = GpTerminal('svc', (None, 'ens'))
    clf2 = GpTerminal('knn', (None, 'ens'))
    clf3 = GpTerminal('dt', (None, 'ens'))

    dummy = GpTerminal('dummy', (None, 'out'))
    
    cEns = FunctionTemplate('_cEns', [TypeArity('ens', (1,1))], 'out')
    
    func_dict = {
            'out' : [ cEns ],
            'ens' : [ boost ]
    }
    
    term_dict = {
            'ens' : [clf1, clf2, clf3],
            'out' : [dummy]
    }

    tree = GpTreeIndividual(gen_full(func_dict, term_dict, 6, 4),0)
    wf = make_workflow(tree, config)

    return tree, wf


if __name__ == "__main__":
    for i in range(0, 5):
        res, wf = gen_trees()

        [print("{}, {}".format(r.name, r.arity)) for r in res.primitives]
        print()
        
        graph.create_graph(res, "tree{}.png".format(i))

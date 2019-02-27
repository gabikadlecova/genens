# -*- coding: utf-8 -*-

"""Test file for GP tree initialization.
"""

from genens.gp.types import FunctionTemplate, TypeArity, GpTerminal
from genens.gp.operators import genFull

def gen_trees():
    
    types = ['out', 'ens', 'data', 'union']
    
    out = []
    
    bag = FunctionTemplate('bagging', None, None,
                            [TypeArity('out', (2,'n'))], 'ens')
    
    boost = FunctionTemplate('boosting', None, None,
                            [TypeArity('out', (2,'n'))], 'ens')
    
    clf1 = GpTerminal('clf1', None, (None, 'ens'))
    clf2 = GpTerminal('clf2', None, (None, 'ens'))
    clf3 = GpTerminal('clf3', None, (None, 'ens'))
    
    
    dummy = GpTerminal('dummy', None, (None, 'out'))
    
    cEns = FunctionTemplate('C', None, None,
                            [TypeArity('ens', (1,1))], 'out')
    
    func_dict = {
            'out' : [ cEns ],
            'ens' : [ bag, boost ]
    }
    
    term_dict = {
            'ens' : [clf1, clf2, clf3],
            'out' : [dummy]
    }
    
    return genFull(func_dict, term_dict, 6, 4)
    
if __name__ == "__main__":
    for i in range(0, 15):
        res = gen_trees()
        [print(r.name) for r in res]
        print()
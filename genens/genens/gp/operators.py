# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution.
"""

import random
from genens.gp.types import GpTreeIndividual


def gen_half(pop_size, config, max_height, max_arity):
    for i in range(0, pop_size):
        yield gen_tree(config.full_config, config.term_config, max_height, max_arity, config.kwargs_config)


def gen_tree(full_dict, term_dict, max_height, max_arity, kwargs_dict):
    tree_list = []
    type_stack = [('out', 1, 1)]
    max_h = 0

    while len(type_stack):
        next_type, ar, h = type_stack.pop()
        max_h = max(h, max_h)

        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))
        
        if h < max_height:
            choose_from = full_dict[next_type]
        else:
            choose_from = term_dict[next_type]
        
        # template of the next primitive
        next_prim_t = random.choice(choose_from)
        
        prim = next_prim_t.create_primitive(h, max_arity, kwargs_dict[next_prim_t.name])
        
        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)
    
    return GpTreeIndividual(list(reversed(tree_list)), max_h)

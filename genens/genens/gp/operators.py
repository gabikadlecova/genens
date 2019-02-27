# -*- coding: utf-8 -*-

"""This file defines genetic operators used in the evolution.
"""
import random

# TODO need a primitive set, templates to create GpPrims

def genRampedHalfNHalf(pop_size, func_dict, term_dict, max_height, max_arity):
    for i in range(0, pop_size / 2):
        yield genGrow(func_dict, term_dict, max_height, max_arity)
    
    for i in range(pop_size / 2, pop_size):
        yield genFull(func_dict, term_dict, max_height, max_arity)


def genGrow(func_dict, term_dict, max_height, max_arity):
    return genTree(func_dict + term_dict, term_dict, max_height, max_arity)


def genFull(func_dict, term_dict, max_height, max_arity):
    return genTree(func_dict, term_dict, max_height, max_arity)


def genTree(full_dict, term_dict, max_height, max_arity):
    tree_list = []
    type_stack = [('out', 1)]
    
    while len(type_stack):
        next_type, h = type_stack.pop()
        
        if h < max_height:
            choose_from = full_dict[next_type]
        else:
            choose_from = term_dict[next_type]
        
        # template of the next primitive
        next_prim_t = random.choice(choose_from)
        
        if next_prim_t.arity > 0:
            for child_type in next_prim_t.node_type[0]:
                name = child_type.get_name()
                type_stack.append((name, h + 1))

        prim = next_prim_t.create_primitive(max_arity)
        tree_list.append(prim)
    
    return tree_list

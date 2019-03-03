# -*- coding: utf-8 -*-

"""This file defines genetic operators used in the evolution.
"""
import random

# TODO need a primitive set, templates to create GpPrims


def gen_half(pop_size, full_dict, grow_dict, term_dict, kwargs_dict, max_height, max_arity):
    # grow
    for i in range(0, pop_size / 2):
        yield gen_tree({**full_dict, **grow_dict}, term_dict, max_height, max_arity, kwargs_dict)

    # full
    for i in range(pop_size / 2, pop_size):
        yield gen_tree(full_dict, term_dict, max_height, max_arity, kwargs_dict)


def gen_tree(full_dict, term_dict, max_height, max_arity, kwargs_dict):
    tree_list = []
    type_stack = [('out', 1, 1)]
    
    while len(type_stack):
        next_type, ar, h = type_stack.pop()
        
        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))
        
        if h < max_height:
            choose_from = full_dict[next_type]
        else:
            choose_from = term_dict[next_type]
        
        # template of the next primitive
        next_prim_t = random.choice(choose_from)
        
        prim = next_prim_t.create_primitive(max_arity, kwargs_dict)
        
        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)
    
    return tree_list

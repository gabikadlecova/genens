# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution.
"""

import random
from genens.gp.types import GpTreeIndividual


def gen_population(pop_size, config, max_height, max_arity):
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


def mutate_subtree(gp_tree, config, max_arity, eps=2):
    """


    :param GpTreeIndividual gp_tree:
    :return:
    """
    mut_end_point = random.randrange(len(gp_tree.primitives))
    root_height = gp_tree.primitives[mut_end_point].height

    mut_begin_point, subtree_height = gp_tree.subtree(mut_end_point)
    new_height = random.randint(1, subtree_height + eps)  # generate a smaller or a bigger subtree

    new_tree = gen_tree(config.full_config, config.term_config, new_height, max_arity, config.kwargs_config)
    for prim in new_tree.primitives:
        prim.height += root_height

    # replace subtree, update height
    gp_tree.primitives[mut_begin_point : (mut_end_point + 1)] = new_tree.primitives
    gp_tree.max_height = max(gp_tree.max_height, root_height + new_tree.max_height)
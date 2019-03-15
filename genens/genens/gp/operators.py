# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution.
"""

import random
from genens.gp.types import GpTreeIndividual


def gen_population(toolbox, pop_size, max_height, max_arity):
    for i in range(0, pop_size):
        # TODO random values of height and arity
        yield toolbox.individual(max_height, max_arity)


def gen_tree(toolbox, config, max_height, max_arity, first_type='out'):
    tree_list = []
    type_stack = [(first_type, 1, 1)]
    max_h = 0

    while len(type_stack):
        next_type, ar, h = type_stack.pop()
        max_h = max(h, max_h)

        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))

        if h < max_height:
            choose_from = config.full_config[next_type]
        else:
            choose_from = config.term_config[next_type]

        # template of the next primitive
        next_prim_t = random.choice(choose_from)

        prim = next_prim_t.create_primitive(h, max_arity, config.kwargs_config[next_prim_t.name])

        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)

    return toolbox.TreeIndividual(list(reversed(tree_list)), max_h)


def mutate_subtree(gp_tree, config, max_arity, eps=2):
    """
    Replaces a random subtree with a new random tree. The height of the generated subtree
    is between 1 and previous subtree height + ``eps``.

    :param GpTreeIndividual gp_tree:
    :param config: Configuration of the evolution.
    :param int max_arity: Maximum arity of a child group (see ``gen_tree``).
    :param int eps:
        Difference between the height of the new subtree and the previous subtree
        must not be greater than this value.
    :return: The mutated tree.
    """
    mut_end_point = random.randrange(len(gp_tree.primitives))
    root_height = gp_tree.primitives[mut_end_point].height

    _, subtree_height = gp_tree.subtree(mut_end_point)
    new_height = random.randint(1, subtree_height + eps)  # generate a smaller or a bigger subtree

    new_tree = gen_tree(config.full_config, config.term_config, new_height,
                        max_arity, config.kwargs_config,
                        first_type=gp_tree.primitives[mut_end_point].out_type)

    off, _ = _swap_subtrees(gp_tree, new_tree, mut_end_point, keep_2=False)

    return off


def crossover_one_point(gp_tree_1, gp_tree_2):
    """


    :param GpTreeIndividual gp_tree_1:
    :param GpTreeIndividual gp_tree_2:
    :return:
    """
    type_set_1 = {gp_ind.out_type for gp_ind in gp_tree_1.primitives}
    type_set_2 = {gp_ind.out_type for gp_ind in gp_tree_2.primitives}

    common_types = list(type_set_1.intersection(type_set_2))
    cx_type = random.choice(common_types)

    eligible_1 = [ind for ind, node in enumerate(gp_tree_1.primitives) if node.out_type == cx_type]
    eligible_2 = [ind for ind, node in enumerate(gp_tree_2.primitives) if node.out_type == cx_type]

    cx_1 = random.choice(eligible_1)
    cx_2 = random.choice(eligible_2)

    return _swap_subtrees(gp_tree_1, gp_tree_2, cx_1, cx_2)


def _swap_subtrees(tree_1, tree_2, ind_1, ind_2, keep_2=True):
    """
    Swaps subtrees of argument trees. Subtree position is determined by
    ``ind_1`` and ``ind_2``.

    If ``keep_2`` is False, the subtree from
    ``tree_1`` is not inserted into ``tree_2`` and only the first tree
    is returned.

    :param GpTreeIndividual tree_1: First tree.
    :param GpTreeIndividual tree_2: Second tree.
    :param int ind_1: Index of the subtree of the first subtree.
    :param int ind_2: Index of the subtree of the second tree.
    :param bool keep_2: Indicates whether the second modified tree would be returned.
    :return (GpTreeIndividual, GpTreeIndividual) or (GpTreeIndividual,):
        Returns both trees with swapped subtrees or only the first tree with
        a subtree inserted from ``tree_2`` (according to ``keep_2``).
    """

    # update node heights
    root_height_1 = tree_1.primitives[ind_1].height
    root_height_2 = tree_1.primitives[ind_2].height
    height_diff = root_height_1 - root_height_2

    def move_node(prim, diff):
        prim.height = prim.height + diff
        return prim

    # subtree indices
    ind_begin_1, _ = tree_1.subtree(ind_1)
    ind_begin_2, _ = tree_1.subtree(ind_2)

    ind_end_1 = ind_1 + 1
    ind_end_2 = ind_2 + 1

    # insert into tree_1
    subtree_2 = (move_node(prim, height_diff)
                 for prim in tree_2.primitives[ind_begin_2 : ind_end_2])

    tree_1.primitives[ind_begin_1 : ind_end_1] = subtree_2
    tree_1.max_height = max(prim.height for prim in tree_1.primitives)  # update height

    # insert into tree_2
    if keep_2:
        subtree_1 = (move_node(prim, -height_diff)
                     for prim in tree_1.primitives[ind_begin_1: ind_end_1])

        tree_2.primitives[ind_begin_2 : ind_end_2] = subtree_1
        tree_2.max_height = max(prim.height for prim in tree_2.primitives)  # update height
        return tree_1, tree_2

    return tree_1,




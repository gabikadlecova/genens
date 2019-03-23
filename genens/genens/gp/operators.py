# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution.
"""

import gc

import random
from genens.gp.types import GpTreeIndividual
from genens.render import graph

from deap import creator, tools

# TODO write docstrings


def gen_population(toolbox):
    # TODO random values of height and ? arity
    return toolbox.individual()  # keyword args max_height and first_type


def gen_tree(config, max_height=None, first_type='out'):
    tree_list = []
    type_stack = [(first_type, 1, 1)]
    tree_height = 0

    if max_height is None:
        max_height = config.max_height

    while len(type_stack):
        next_type, ar, h = type_stack.pop()
        tree_height = max(h, tree_height)

        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))

        if h < max_height:
            choose_from = config.full_config[next_type]
        else:
            choose_from = config.term_config[next_type]

        # template of the next primitive
        next_prim_t = random.choice(choose_from)

        prim = next_prim_t.create_primitive(h, config.max_arity,
                                            config.kwargs_config[next_prim_t.name])

        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)

    return creator.TreeIndividual(list(reversed(tree_list)), tree_height)


def mutate_subtree(toolbox, gp_tree, eps=4):
    """
    Replaces a random subtree with a new random tree. The height of the generated subtree
    is between 1 and previous subtree height + ``eps``.

    :param toolbox: Toolbox of the genetic algorithm.
    :param GpTreeIndividual gp_tree:
    :param int eps:
        Difference between the height of the new subtree and the previous subtree
        must not be greater than this value.
    :return: The mutated tree.
    """
    if len(gp_tree.primitives) < 2:
        return gp_tree

    mut_end_point = random.randrange(len(gp_tree.primitives) - 1)

    _, subtree_height = gp_tree.subtree(mut_end_point)
    new_height = random.randint(1, subtree_height + eps)  # generate a smaller or a bigger subtree

    new_tree = toolbox.individual(max_height=new_height,
                                  first_type=gp_tree.primitives[mut_end_point].out_type)
    new_root_point = len(new_tree.primitives) - 1

    offs = _swap_subtrees(gp_tree, new_tree, mut_end_point, new_root_point, keep_2=False)

    return offs[0]


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
    root_height_2 = tree_2.primitives[ind_2].height
    height_diff = root_height_1 - root_height_2

    def move_node(prim, diff):
        prim.height = prim.height + diff
        return prim

    # subtree indices
    ind_begin_1, _ = tree_1.subtree(ind_1)
    ind_begin_2, _ = tree_2.subtree(ind_2)

    ind_end_1 = ind_1 + 1
    ind_end_2 = ind_2 + 1

    # insert into tree_1 - copy subtree from tree_2
    subtree_2 = [move_node(prim, height_diff)
                 for prim in tree_2.primitives[ind_begin_2 : ind_end_2]]

    # insert into tree_2
    if keep_2:
        subtree_1 = (move_node(prim, -height_diff)
                     for prim in tree_1.primitives[ind_begin_1 : ind_end_1])

        tree_2.primitives[ind_begin_2 : ind_end_2] = subtree_1
        tree_2.max_height = max(prim.height for prim in tree_2.primitives)  # update height

        # TODO remove
        tree_2.validate_tree()

    # insert into tree_1 - insert subtree
    tree_1.primitives[ind_begin_1: ind_end_1] = subtree_2
    tree_1.max_height = max(prim.height for prim in tree_1.primitives)  # update height

    # TODO remove
    tree_1.validate_tree()

    return tree_1, tree_2 if keep_2 else tree_1,


def gen_valid(toolbox, timeout=1000):
    i = 0

    while True:
        if i >= timeout:
            raise ValueError("Couldn't generate a valid individual.")  # TODO specific

        ind = toolbox.individual()
        score = toolbox.evaluate(ind)

        if score is not None:
            ind.fitness.values = score
            return ind

        i += 1


def ea_run(population, toolbox, n_gen, pop_size, cx_pb, mut_pb):
    scores = toolbox.map(toolbox.evaluate, population)

    for ind, score in zip(population, scores):
        if score is None:
            continue

        ind.fitness.values = score

    # remove individuals which threw exceptions
    population[:] = [ind for ind in population if ind.fitness.valid]

    for i in range(pop_size - len(population)):
        population.append(gen_valid(toolbox))

    toolbox.log(population, 0)

    population[:] = toolbox.select(population, pop_size)  # assigns crowding distance

    for g in range(n_gen):

        print("Gen {}".format(g))

        population[:] = tools.selTournamentDCD(population, pop_size)
        offspring = list(toolbox.map(toolbox.clone, population))

        for ch1, ch2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.cx_one_point(ch1, ch2)
                del ch1.fitness.values
                del ch2.fitness.values
                # mutation

        for mut in offspring:
            if random.random() < mut_pb:
                toolbox.mutate_subtree(mut)
                del mut.fitness.values

        offs_to_eval = [ind for ind in offspring if not ind.fitness.valid]

        scores = toolbox.map(toolbox.evaluate, offs_to_eval)
        for off, score in zip(offs_to_eval, scores):
            if score is None:
                continue

            off.fitness.values = score

        # remove offspring which threw exceptions
        offspring[:] = [ind for ind in offspring if ind.fitness.valid]

        for i in range(pop_size - len(offspring)):
            offspring.append(gen_valid(toolbox))

        population[:] = toolbox.select(population + offspring, pop_size)

        toolbox.log(population, g)

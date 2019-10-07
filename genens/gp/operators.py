# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution as well as the main loop
of the algorithm and initialization methods.
"""

from deap import tools
from itertools import chain
from joblib import Parallel, delayed

from ..gp.types import GpTreeIndividual, GpFunctionTemplate, DeapTreeIndividual

import math
import numpy as np
import random


def gen_individual(toolbox, config, out_type='out'):
    """
    Generates a random tree individual using the toolbox and arity and height configuration.
    The ``ou_type`` parameter specifies the output type of the root of the tree.

    :param toolbox: Toolbox which contains methods to create the individual.
    :param GenensConfig config: Configuration of Genens
    :param out_type: Output type of the root of the tree individual.
    :return: A new random tree individual.
    """
    arity = random.randint(2, config.max_arity)
    height = random.randint(1, config.max_height)

    return toolbox.individual(max_height=height, max_arity=arity, first_type=out_type)


def _choose_group_weighted(config, groups):
    # weighted choice
    total_sum = np.sum((config.group_weights[group] for group in groups))
    rand_val = random.random() * total_sum

    # determine which group was chosen
    group_chosen = None
    partial_sum = 0.0
    for group in groups:
        partial_sum += config.group_weights[group]
        if partial_sum > rand_val:
            return group

    if group_chosen is None:
        raise RuntimeError("Invalid weight sum.")  # should never get here


def choose_prim(config, prim_list, weighted=True, use_groups=True):
    """
    Performs a selection from list of primitives.

    :param GenensConfig config: Configuration of Genens which contains the weight configuration.
    :param list prim_list: List of primitives to choose from.
    :param bool weighted: Determines whether the selection is weighted (according to primitive group weights).
    :param bool use_groups: If false, primitive groups are ignored
    :return GpPrimitive: The selected primitive.
    """
    if not use_groups:
        return random.choice(prim_list)

    group_names = {prim.group for prim in prim_list}  # primitive groups to choose from

    if weighted:
        group_chosen = _choose_group_weighted(config, group_names)
    else:
        group_chosen = random.choice(group_names)

    prim_possible = [prim for prim in prim_list if prim.group == group_chosen]
    return random.choice(prim_possible)


def gen_tree(config, max_height=None, max_arity=None, first_type='out', weighted=True, use_groups=True):
    """
    Creates a random tree individual to be used in the DEAP framework.

    :param config: Configuration of nodes, keyword arguments and arity and height limits.
    :param max_height: Height limit of the tree; if not specified, the configuration limit is used.
    :param max_arity: Artity limit of function nodes; if not specified, the configuration limit is used.
    :param first_type: Output type of the root of the tree.
    :param weighted: Determines whether the node selection is weighted.
    :param bool use_groups: If false, primitive groups are ignored
    :return: A random tree individual.
    """
    tree_list = []
    type_stack = [(first_type, 1, 1)]
    tree_height = 0

    if max_height is None:
        max_height = config.max_height

    if max_arity is None:
        max_arity = config.max_arity

    while len(type_stack):
        next_type, ar, h = type_stack.pop()
        tree_height = max(h, tree_height)

        # more children should be generated
        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))

        # choose only terminals in the last level
        if h < max_height:
            choose_from = config.full_config[next_type]
        else:
            choose_from = config.term_config[next_type]

        # template of the next primitive
        next_prim_t = choose_prim(config, choose_from, weighted=weighted, use_groups=use_groups)

        prim = next_prim_t.create_primitive(h, max_arity,
                                            config.kwargs_config[next_prim_t.name])

        # append child types and arities to the stack
        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)

    return DeapTreeIndividual(list(reversed(tree_list)), tree_height)


def mutate_subtree(toolbox, gp_tree, eps=0.2):
    """
    Replaces a randomly chosen subtree with a new random tree. The height of the generated subtree
    is between 1 and previous subtree height + ``eps``.

    :param toolbox: Toolbox of the genetic algorithm.
    :param GpTreeIndividual gp_tree:
    :param float eps:
        Height of the new subtree lies in the interval ((1-eps) * old_height, (1+eps) * old_height).

    :return: The mutated tree.
    """
    # mutation does not replace the whole tree
    if len(gp_tree.primitives) < 2:
        return gp_tree

    # select a node other than the root
    mut_end_point = random.randrange(len(gp_tree.primitives) - 1)

    _, subtree_height = gp_tree.subtree(mut_end_point)

    lower = math.floor((1.0 - eps) * subtree_height)
    upper = math.ceil((1.0 + eps) * subtree_height)

    new_height = random.randint(lower, upper + 1)  # generate a smaller or a bigger subtree
    new_tree = toolbox.individual(max_height=new_height,
                                  first_type=gp_tree.primitives[mut_end_point].out_type)
    new_root_point = len(new_tree.primitives) - 1

    offs = _swap_subtrees(gp_tree, new_tree, mut_end_point, new_root_point, keep_2=False)

    return offs[0]


def _node_types_match(node, node_template):
    """
    Verifies whether input and output types of two nodes match. The first has fixed arities of
    the input type whereas the second is a template with variable arities.

    :param node: Node with fixed input types arities.
    :param node_template: A node template with variable arities.

    :return bool: True if the types of the nodes match.
    """
    # out types match?
    if node.node_type[1] != node_template.out_type:
        return False

    # in types (and arities) match?
    in_types = node.node_type[0]
    template_types = node_template.type_arities

    if len(in_types) != len(template_types):
        return False

    for in_type_1, in_type_2 in zip(in_types, template_types):
        # check type compatibility
        if in_type_1.name != in_type_2.prim_type:
            return False

        # check arity compatibility
        if not in_type_2.is_valid_arity(in_type_1.arity):
            return False

    return True


def mutate_node_swap(config, gp_tree):
    """
    Mutates randomly a node of the tree --- replaces it with a new node with the same type.

    :param GenensConfig config: Configuration which contains node definitions.
    :param GpTreeIndividual gp_tree: Individual to be mutated
    :return: Mutated individual.
    """

    swap_ind = random.randrange(len(gp_tree.primitives))
    swap_node = gp_tree.primitives[swap_ind]

    out_type = swap_node.node_type[1]

    possible_templates = [tm for tm in config.full_config[out_type]
                          if _node_types_match(swap_node, tm)]

    possible_templates += [tm for tm in config.term_config[out_type]
                           if _node_types_match(swap_node, tm)]

    chosen_tm = random.choice(possible_templates)

    if isinstance(chosen_tm, GpFunctionTemplate):
        new_node = chosen_tm.create_primitive(swap_node.height, config.max_arity,
                                              config.kwargs_config[chosen_tm.name],
                                              in_type=swap_node.node_type[0])
    else:
        new_node = chosen_tm.create_primitive(swap_node.height, config.max_arity,
                                              config.kwargs_config[chosen_tm.name])

    gp_tree.primitives[swap_ind] = new_node
    return gp_tree


def mutate_node_args(toolbox, config, gp_tree, hc_repeat=0, keep_last=False):
    """
    Mutates a random argument of a node from the GP tree. If ``hc_repeat`` is greater
    than zero, performs a hill-climbing mutation of the argument.

    :param toolbox: GP toolbox.
    :param GenensConfig config: Configuration of the evolution.
    :param DeapTreeIndividual gp_tree: Individual to be mutated.
    :param int hc_repeat:
    If equal to n = 0, mutates a single argument and returns the mutated individual.
    If equal to n > 0, performs a hill-climbing mutation of n iterations, keeping the best
    individual.

    :param bool keep_last:
    If True, returns the last mutant even if the best individual was the original individual.

    :return: The mutated individual.
    """
    mut_ind = random.randint(0, len(gp_tree.primitives) - 1)

    mut_node = gp_tree.primitives[mut_ind]

    # no parameters to mutate
    if not len(mut_node.obj_kwargs):
        return gp_tree

    mut_arg = random.choice(list(mut_node.obj_kwargs.keys()))

    # mutate one parameter, do not perform hillclimbing
    if hc_repeat < 1:
        _mut_args(config, mut_node, mut_arg)
        return gp_tree

    # hill-climbing initial fitness
    if not gp_tree.fitness.valid:
        score = toolbox.evaluate(gp_tree)

        # mutate only valid individuals
        if score is None:
            return gp_tree
        gp_tree.fitness.values = score

    # hill-climbing procedure
    has_mutated = False
    for i in range(hc_repeat):
        mutant = toolbox.clone(gp_tree)
        mut_node = mutant.primitives[mut_ind]

        _mut_args(config, mut_node, mut_arg)

        score = toolbox.evaluate(mutant)
        # skip invalid mutants
        if score is None:
            continue

        # the mutant is better, keep it
        if score >= gp_tree.fitness.values:
            gp_tree.primitives[mut_ind] = mut_node  # copy value to the mutated tree
            gp_tree.fitness.values = score

            has_mutated = True
        else:
            # return the last one if no mutant was better than the first individual
            if keep_last and i == hc_repeat - 1 and not has_mutated:
                gp_tree.primitives[ mut_ind] = mut_node
                gp_tree.fitness.values = score

    return gp_tree


def _mut_args(config, node, key):
    """
    Mutates a keyword argument of a single node.

    :param GenensConfig config: Configuration of Genens which contains lists of possible keyword arguments values.
    :param node: Node to be mutated.
    :param key: Key of the argument which will be mutated
    """

    node.obj_kwargs[key] = random.choice(config.kwargs_config[node.name][key])


def crossover_one_point(gp_tree_1, gp_tree_2):
    """
    Performs a crossover of two tree individuals.

    :param DeapTreeIndividual gp_tree_1: First individual.
    :param DeapTreeIndividual gp_tree_2: Second individual.
    :return: Tree individuals modified by the crossover.
    """
    type_set_1 = {gp_ind.out_type for gp_ind in gp_tree_1.primitives}
    type_set_2 = {gp_ind.out_type for gp_ind in gp_tree_2.primitives}

    common_types = list(type_set_1.intersection(type_set_2))
    cx_type = random.choice(common_types)

    # nodes which can be roots of subtrees to be swapped
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


def gen_valid(toolbox, timeout=100):
    """
    Tries to generate an individual with a valid items. Raises an error if it does
    not succeed in ``timeout`` iterations.

    :param toolbox: Toolbox with initialization and evaluation methods.
    :param timeout: Number of iterations to perform.
    :return: A valid individual.
    """
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


def _perform_cx(cx_func, cx_pb, ch1, ch2):
    if random.random() < cx_pb:
        ch1, ch2 = cx_func(ch1, ch2)
        ch1.reset()
        ch2.reset()

    return ch1, ch2


def _perform_mut(mut_func, mut_pb, mut):
    if random.random() < mut_pb:
        mut = mut_func(mut)
        mut.reset()

    return mut


def ea_run(population, toolbox, n_gen, pop_size, cx_pb, mut_pb, mut_args_pb, mut_node_pb, n_jobs=1):
    """
    Performs a run of the evolutionary algorithm.

    :param population: Initial population of individuals.
    :param toolbox: Toolbox with methods to use in the evolution.
    :param n_gen: Number of generations.
    :param pop_size: Population size.
    :param cx_pb:
    :param mut_pb:
    :param mut_args_pb:
    :param mut_node_pb:
    :param n_jobs: Number of jobs for multiprocessing
                   ``n_jobs = k, k processors are used
                   ``n_jobs`` = 1, multiprocessing is not used
                   ``n_jobs = k, k + 1 processors are used - for -1, all processors are used
    """

    # TODO later change to verbose
    print('Initial population generated.')

    # evaluate first gen 
    with Parallel(n_jobs=n_jobs) as parallel:
        scores = toolbox.map(toolbox.evaluate, population, parallel=parallel)

    for ind, score in zip(population, scores):
        if score is None:
            continue

        ind.fitness.values = score

    # remove individuals which threw exceptions and generate new valid individuals
    population[:] = [ind for ind in population if ind.fitness.valid]

    valid = Parallel(n_jobs=n_jobs)(delayed(gen_valid)(toolbox) for i in range(pop_size - len(population)))
    population += valid

    toolbox.log(population, 0)

    population[:] = toolbox.select(population, pop_size)  # assigns crowding distance

    for g in range(n_gen):
        print("Gen {}".format(g))
        toolbox.next_gen()

        # selection for operations
        population[:] = tools.selTournamentDCD(population, pop_size)
        offspring = toolbox.map(toolbox.clone, population)

        with Parallel(n_jobs=n_jobs) as parallel:
            # crossover - subtree
            offspring = parallel(delayed(_perform_cx)(toolbox.cx_one_point, cx_pb, ch1, ch2)
                                 for ch1, ch2 in zip(offspring[::2], offspring[1::2]))
            offspring = list(chain.from_iterable(offspring))  # chain cx tuple elements

            # mutation - subtree
            offspring = parallel(delayed(_perform_mut)(toolbox.mutate_subtree, mut_pb, mut)
                                 for mut in offspring)

            # mutation - node args
            offspring = parallel(delayed(_perform_mut)(toolbox.mutate_node_args, mut_args_pb, mut)
                                 for mut in offspring)

            # mutation - node swap
            offspring = parallel(delayed(_perform_mut)(toolbox.mutate_node_swap, mut_node_pb, mut)
                                 for mut in offspring)

            offs_to_eval = [ind for ind in offspring if not ind.fitness.valid]

            # evaluation of changed offspring
            scores = toolbox.map(toolbox.evaluate, offs_to_eval, parallel=parallel)
            for off, score in zip(offs_to_eval, scores):
                if score is None:
                    continue

                off.fitness.values = score

            # remove offspring which threw exceptions
            offspring[:] = [ind for ind in offspring if ind.fitness.valid]

            valid = parallel(delayed(gen_valid)(toolbox) for i in range(pop_size - len(offspring)))
            offspring += valid

        population[:] = toolbox.select(population + offspring, pop_size)

        toolbox.log(population, g + 1)

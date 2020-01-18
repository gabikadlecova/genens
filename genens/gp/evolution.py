# -*- coding: utf-8 -*-

"""
This module defines genetic operators used in the evolution as well as the main loop
of the algorithm and initialization methods.
"""

import logging
import random

from deap import tools
from functools import partial
from itertools import chain
from joblib import Parallel, delayed

from genens.log_utils import set_log_handler
from genens.render.graph import tree_str


def gen_individual(toolbox, config, out_type='out'):
    """
    Generates a random tree individual using the toolbox and arity and height configuration.
    The ``ou_type`` parameter specifies the output type of the root of the tree.

    :param toolbox: Toolbox which contains methods to create the individual.
    :param GenensConfig config: Configuration of Genens
    :param out_type: Output type of the root of the tree individual.
    :return: A new random tree individual.
    """
    arity = random.randint(config.min_arity, config.max_arity)  # randint is inclusive for both limits
    height = random.randint(config.min_height, config.max_height)

    return toolbox.individual(max_height=height, max_arity=arity, first_type=out_type)


@set_log_handler
def gen_valid(toolbox, timeout=100):
    """
    Tries to generate an individual with a valid items. Raises an error if it does
    not succeed in ``timeout`` iterations.

    :param toolbox: Toolbox with initialization and evaluation methods.
    :param timeout: Number of iterations to perform.
    :return: A valid individual.
    """

    for i in range(timeout):
        ind = toolbox.individual()
        score = toolbox.evaluate(ind)

        logger = logging.getLogger("genens")
        logger.debug(f"Generate valid ({i}/{timeout}):\n {tree_str(ind, with_hyperparams=True)}")

        if score is not None:
            ind.fitness.values = score
            return ind

    raise ValueError("Couldn't generate a valid individual.")  # TODO specific


@set_log_handler
def _perform_cx(cx_func, cx_pb, ch1, ch2, **kwargs):
    parent1_str = tree_str(ch1)
    parent2_str = tree_str(ch2)

    if random.random() < cx_pb:
        ch1, ch2 = cx_func(ch1, ch2, **kwargs)
        ch1.reset()
        ch2.reset()

        logger = logging.getLogger("genens")
        logger.debug(f"Crossover ({cx_func}):\n {parent1_str}\n x \n{parent2_str}\n ->"
                     f"\nCh1:\n{tree_str(ch1)}\nCh2:\n{tree_str(ch2)}")

    return ch1, ch2


@set_log_handler
def _perform_mut(mut_func, mut_pb, mut, **kwargs):
    parent_str = tree_str(mut, with_hyperparams=True)

    if random.random() < mut_pb:
        mut = mut_func(mut, **kwargs)
        mut.reset()

        logger = logging.getLogger("genens")
        logger.debug(f"Mutation ({mut_func}):\n {parent_str}\n -> \n{tree_str(mut, with_hyperparams=True)}")

    return mut


@set_log_handler
def _perform_eval(eval_func, ind):
    return eval_func(ind)


def ea_run(population, toolbox, n_gen, pop_size, cx_pb, mut_pb, mut_args_pb, mut_node_pb, n_jobs=1,
           min_large_tree_height=3, verbose=1):
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

    :param  min_large_tree_height: Perform structural changes only on trees of at least this height.
    :param verbose: Print verbosity
    """

    evaluate_func = partial(_perform_eval, toolbox.evaluate, log_setup=toolbox.log_setup)

    if verbose >= 1:
        print('Initial population generated.')

    logger = logging.getLogger("genens")
    for i, ind in enumerate(population):
        logger.debug(f"Generation 0, individual {i}:\n{tree_str(ind, with_hyperparams=True)}")

    # evaluate first gen 
    with Parallel(n_jobs=n_jobs) as parallel:
        scores = toolbox.map(evaluate_func, population, parallel=parallel)

    for ind, score in zip(population, scores):
        if score is None:
            continue

        ind.fitness.values = score

    if verbose >= 1:
        print("First gen evaluated")

    # remove individuals which threw exceptions and generate new valid individuals
    population[:] = [ind for ind in population if ind.fitness.valid]
    valid = Parallel(n_jobs=n_jobs)(delayed(gen_valid)(toolbox, log_setup=toolbox.log_setup)
                                    for _ in range(pop_size - len(population)))
    population += valid

    toolbox.log(population, 0)
    population[:] = toolbox.select(population, pop_size)  # assigns crowding distance

    for g in range(n_gen):
        if verbose >= 1:
            print("Gen {}".format(g))

        for i, ind in enumerate(population):
            logger.debug(f"Generation {g}, individual {i}:\n{tree_str(ind, with_hyperparams=True)}")

        toolbox.next_gen()

        # selection for operations
        population[:] = tools.selTournamentDCD(population, pop_size)
        larger_trees = [tree for tree in population if tree.max_height >= min_large_tree_height]
        all_offspring = []

        with Parallel(n_jobs=n_jobs) as parallel:
            if verbose >= 2:
                print(f"Gen {g} - crossover")

            # crossover - subtree
            offspring = toolbox.map(toolbox.clone, larger_trees)
            offspring = parallel(
                delayed(_perform_cx)(
                    toolbox.cx_one_point,  # func
                    cx_pb, ch1, ch2, min_node_depth=min_large_tree_height - 1, log_setup=toolbox.log_setup  # args
                )
                for ch1, ch2 in zip(offspring[::2], offspring[1::2])
            )
            offspring = list(chain.from_iterable(offspring))  # chain cx tuples to list of offspring
            all_offspring += offspring

            if verbose >= 2:
                print(f"Gen {g} - mutation")

            # mutation - subtree
            offspring = toolbox.map(toolbox.clone, larger_trees)
            offspring = parallel(
                delayed(_perform_mut)(
                    toolbox.mutate_subtree,  # func
                    mut_pb, mut, log_setup=toolbox.log_setup  # args
                )
                for mut in offspring
            )
            all_offspring += offspring

            # mutation - node swap
            offspring = toolbox.map(toolbox.clone, population)
            offspring = parallel(
                delayed(_perform_mut)(
                    toolbox.mutate_node_swap,  # func
                    mut_node_pb, mut, log_setup=toolbox.log_setup  # args
                )
                for mut in offspring
            )
            all_offspring += offspring

            # for large new offspring, perform hc to make them competitive
            for offs in all_offspring:
                if offs.height < min_large_tree_height:
                    continue

                toolbox.gradual_hillclimbing(offs)

            # ----------------

            # mutation - node args
            offspring = toolbox.map(toolbox.clone, population)
            offspring = parallel(
                delayed(_perform_mut)(
                    toolbox.mutate_args,  # func
                    mut_args_pb, mut, log_setup=toolbox.log_setup  # args
                )
                for mut in offspring
            )
            all_offspring += offspring

            offs_to_eval = [ind for ind in all_offspring if not ind.fitness.valid]

            # evaluation of changed offspring
            scores = toolbox.map(evaluate_func, offs_to_eval, parallel=parallel)
            for off, score in zip(offs_to_eval, scores):
                if score is None:
                    continue

                off.fitness.values = score

            # remove offspring which threw exceptions
            all_offspring[:] = [ind for ind in all_offspring if ind.fitness.valid]
            valid = parallel(delayed(gen_valid)(toolbox, log_setup=toolbox.log_setup)
                             for _ in range(pop_size - len(all_offspring)))
            all_offspring += valid

        population[:] = toolbox.select(population + all_offspring, pop_size)

        toolbox.log(population, g + 1)

    if verbose >= 1:
        print("Evolution completed")

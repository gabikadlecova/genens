import logging
import random
from genens.gp.tree import swap_subtrees


def crossover_one_point(gp_tree_1, gp_tree_2, min_node_depth=0):
    """
    Performs a crossover of two tree individuals.

    :param DeapTreeIndividual gp_tree_1: First individual.
    :param DeapTreeIndividual gp_tree_2: Second individual.
    :param int min_node_depth: Minimal crossover point depth in both trees.
    :return: Tree individuals modified by the crossover.
    """
    type_set_1 = {gp_ind.out_type for gp_ind in gp_tree_1.primitives if gp_ind.depth >= min_node_depth}
    type_set_2 = {gp_ind.out_type for gp_ind in gp_tree_2.primitives if gp_ind.depth >= min_node_depth}

    common_types = list(type_set_1.intersection(type_set_2))
    if not len(common_types):
        logger = logging.getLogger("genens")
        logger.debug(f"Crossover - no nodes of the same type:\n{gp_tree_1}\nx\n{gp_tree_2}")
        return gp_tree_1, gp_tree_2

    cx_type = random.choice(common_types)

    # nodes which can be roots of subtrees to be swapped
    eligible_1 = [ind for ind, node in enumerate(gp_tree_1.primitives)
                  if node.out_type == cx_type and node.depth >= min_node_depth]
    eligible_2 = [ind for ind, node in enumerate(gp_tree_2.primitives)
                  if node.out_type == cx_type and node.depth >= min_node_depth]

    cx_1 = random.choice(eligible_1)
    cx_2 = random.choice(eligible_2)

    return swap_subtrees(gp_tree_1, gp_tree_2, cx_1, cx_2)

import random
from genens.gp.tree import swap_subtrees


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

    return swap_subtrees(gp_tree_1, gp_tree_2, cx_1, cx_2)

import numpy as np
import random

from genens.gp.types import DeapTreeIndividual


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

    group_names = list({prim.group for prim in prim_list})  # primitive groups to choose from

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
    :param max_arity: Arity limit of function nodes; if not specified, the configuration limit is used.
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
        node_depth = h - 1
        tree_height = max(h, tree_height)

        # more children should be generated
        if ar - 1 > 0:
            type_stack.append((next_type, ar - 1, h))

        # choose only terminals in the last level
        if h < max_height - 1:
            choose_from = config.full_config[next_type]
        else:
            choose_from = config.term_config[next_type]

        # template of the next primitive
        next_prim_t = choose_prim(config, choose_from, weighted=weighted, use_groups=use_groups)
        prim = next_prim_t.create_primitive(node_depth, max_arity,
                                            config.kwargs_config[next_prim_t.name])

        # append child types and arities to the stack
        if prim.arity > 0:
            for child_type in prim.node_type[0]:
                type_stack.append((child_type.name, child_type.arity, h + 1))

        tree_list.append(prim)

    return DeapTreeIndividual(list(reversed(tree_list)), tree_height)


def swap_subtrees(tree_1, tree_2, ind_1, ind_2, keep_2=True):
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
    root_height_1 = tree_1.primitives[ind_1].depth
    root_height_2 = tree_2.primitives[ind_2].depth
    height_diff = root_height_1 - root_height_2

    def move_node(prim, diff):
        prim.height = prim.depth + diff
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
        tree_2.max_height = max(prim.depth + 1 for prim in tree_2.primitives)  # update height

    # insert into tree_1 - insert subtree
    tree_1.primitives[ind_begin_1: ind_end_1] = subtree_2
    tree_1.max_height = max(prim.depth + 1 for prim in tree_1.primitives)  # update height

    return tree_1, tree_2 if keep_2 else tree_1,

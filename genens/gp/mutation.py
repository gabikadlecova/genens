import logging
import math
import random

from genens.gp.tree import swap_subtrees
from genens.gp.types import GpFunctionTemplate


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

    offs = swap_subtrees(gp_tree, new_tree, mut_end_point, new_root_point, keep_2=False)

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


def mutate_args(config, gp_tree, multiple_nodes=False, multiple_args=False):
    # choose only nodes with hyperparameters
    prims = [prim for prim in gp_tree.primitives if len(prim.obj_kwargs)]
    if not len(prims):
        return gp_tree

    if multiple_nodes:
        mut_inds = random.sample([i for i in range(len(prims))])
        for ind in mut_inds:
            _mutate_node_args(config, prims[ind], multiple=multiple_args)

    else:
        mut_node = random.choice(prims)
        _mutate_node_args(config, mut_node, multiple=multiple_args)

    return gp_tree


def _mutate_node_args(config, mut_node, multiple=False):
    # no parameters to mutate
    if not len(mut_node.obj_kwargs):
        return mut_node

    all_keys = list(mut_node.obj_kwargs.keys())

    if multiple:
        n_kwargs = random.randint(1, len(all_keys))
        mut_kwargs = random.sample(all_keys, n_kwargs)

        for key in mut_kwargs:
            _mut_arg(config, mut_node, key)
    else:
        mut_arg = random.choice(all_keys)
        _mut_arg(config, mut_node, mut_arg)

    return mut_node


def perform_hillclimbing(toolbox, gp_tree, mut_func,
                         hc_repeat=1, keep_last=False):
    # hill-climbing initial fitness
    if not gp_tree.fitness.valid:
        score = toolbox.evaluate(gp_tree)

        # mutate only valid individuals
        if score is None:
            return gp_tree
        gp_tree.fitness.values = score

    logger = logging.getLogger("genens")

    # hill-climbing procedure
    has_mutated = False
    for i in range(hc_repeat):
        mutant = toolbox.clone(gp_tree)
        mut_func(mutant)

        score = toolbox.evaluate(mutant)
        # skip invalid mutants
        if score is None:
            continue

        # return the last one if no mutant was better than the first individual
        should_keep_last = keep_last and i == hc_repeat - 1 and not has_mutated

        # the mutant is better, keep it
        if score >= gp_tree.fitness.values or should_keep_last:
            logger.debug(f"Hillclimbing {i}/{hc_repeat}: score {gp_tree.fitness.values} -> {score},"
                         f"\n{gp_tree}\n>\n{mutant}")

            gp_tree = mutant
            has_mutated = True

    return gp_tree


def _mut_arg(config, node, key):
    """
    Mutates a keyword argument of a single node.

    :param GenensConfig config: Configuration of Genens which contains lists of possible keyword arguments values.
    :param node: Node to be mutated.
    :param key: Key of the argument which will be mutated
    """

    node.obj_kwargs[key] = random.choice(config.kwargs_config[node.name][key])

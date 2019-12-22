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
        _mut_arg(config, mut_node, mut_arg)
        return gp_tree

    gp_tree = _arg_hillclimbing(toolbox, config, gp_tree, mut_ind, mut_arg, hc_repeat=hc_repeat, keep_last=keep_last)
    return gp_tree


def _arg_hillclimbing(toolbox, config, gp_tree, mut_ind, mut_arg, hc_repeat=1, keep_last=False):
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

        _mut_arg(config, mut_node, mut_arg)

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
                gp_tree.primitives[mut_ind] = mut_node
                gp_tree.fitness.values = score

    return gp_tree


# TODO mutate
#   1) single node, one arg
#   2) single node, multiple args
#   (3) multiple nodes, single/multiple args)
#   4) reset args

def mutate_single_arg(config, gp_tree):
    mut_ind = random.randint(0, len(gp_tree.primitives) - 1)
    mut_node = gp_tree.primitives[mut_ind]

    # no parameters to mutate
    if not len(mut_node.obj_kwargs):
        return gp_tree

    mut_arg = random.choice(list(mut_node.obj_kwargs.keys()))
    _mut_arg(config, mut_node, mut_arg)
    return gp_tree


def mutate_multiple_args(config, gp_tree):
    mut_inds = random.sample([i for i in range(len(gp_tree))])

    # no parameters to mutate
    if not len(mut_node.obj_kwargs):
        return gp_tree

    mut_arg = random.choice(list(mut_node.obj_kwargs.keys()))
    _mut_arg(config, mut_node, mut_arg)
    return gp_tree


def _mut_random_arg(config, gp_tree):



def _perform_hillclimbing(toolbox, gp_tree, mut_func,
                          hc_repeat=1, keep_last=False):
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
        mut_func(mutant)

        score = toolbox.evaluate(mutant)
        # skip invalid mutants
        if score is None:
            continue

        # return the last one if no mutant was better than the first individual
        should_keep_last = keep_last and i == hc_repeat - 1 and not has_mutated

        # the mutant is better, keep it
        if score >= gp_tree.fitness.values or should_keep_last:
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

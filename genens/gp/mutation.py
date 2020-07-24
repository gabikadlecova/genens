import logging
import math
import random
from functools import partial

from genens.gp.tree import swap_subtrees
from genens.gp.types import GpFunctionTemplate, GpTreeIndividual


def mutate_subtree(toolbox, gp_tree, eps=0.2, min_node_depth=1):
    """
    Replaces a randomly chosen subtree with a new random tree. The height of the generated subtree
    is between 1 and previous subtree height + ``eps``.

    :param toolbox: Toolbox of the genetic algorithm.
    :param GpTreeIndividual gp_tree:
    :param float eps:
        Height of the new subtree lies in the interval ((1-eps) * old_height, (1+eps) * old_height).

    :param int min_node_depth: Minimal mutation point depth in both trees.

    :return: The mutated tree.
    """
    # mutation does not replace the whole tree
    if gp_tree.max_height < 2:
        return gp_tree

    # select a node other than the root
    eligible_inds = [ind for ind, node in enumerate(gp_tree.primitives) if node.depth >= min_node_depth]
    if not len(eligible_inds):
        logger = logging.getLogger("genens")
        logger.debug(f"Mutation (subtree) - tree too small:\n{gp_tree}")
        return gp_tree

    mut_end_point = random.choice(eligible_inds)
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
    if node.out_type != node_template.out_type:
        return False

    # in types (and arities) match?
    in_types = node.in_type
    template_types = node_template.type_arity_template

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

    out_type = swap_node.out_type

    possible_templates = [tm for tm in config.full_config[out_type]
                          if _node_types_match(swap_node, tm)]

    possible_templates += [tm for tm in config.term_config[out_type]
                           if _node_types_match(swap_node, tm)]

    chosen_tm = random.choice(possible_templates)

    if isinstance(chosen_tm, GpFunctionTemplate):
        new_node = chosen_tm.create_primitive(swap_node.depth, config.max_arity,
                                              config.kwargs_config[chosen_tm.name],
                                              in_type=swap_node.in_type)
    else:
        new_node = chosen_tm.create_primitive(swap_node.depth, config.max_arity,
                                              config.kwargs_config[chosen_tm.name])

    gp_tree.primitives[swap_ind] = new_node
    return gp_tree


def mutate_args(config, gp_tree, multiple_nodes=False, multiple_args=False):
    _, mut_nodes = _get_nodes_for_mut(gp_tree, multiple_nodes=multiple_nodes)
    if not len(mut_nodes):
        return gp_tree

    for node in mut_nodes:
        _mutate_node_args(config, node, multiple=multiple_args)

    return gp_tree


def _get_nodes_for_mut(gp_tree, multiple_nodes=False):
    prim_inds = [i for i, prim in enumerate(gp_tree.primitives) if len(prim.obj_kwargs)]
    if not len(prim_inds):
        return [], []

    if multiple_nodes:
        n_nodes = random.randint(1, len(prim_inds))
    else:
        n_nodes = 1

    ind_sample = random.sample(prim_inds, n_nodes)
    return ind_sample, [gp_tree.primitives[i] for i in ind_sample]


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


def mutate_gradual_hillclimbing(toolbox, config, gp_tree: GpTreeIndividual, multiple_args=True,
                                hc_repeat=5, n_nodes=3, keep_last=False):
    def mut_ind(ind, gp_tree):
        _mutate_node_args(config, gp_tree.primitives[ind], multiple=multiple_args)

    mut_inds, _ = _get_nodes_for_mut(gp_tree, multiple_nodes=True)
    if not len(mut_inds):
        return gp_tree

    n_nodes = min(n_nodes, len(mut_inds))
    mut_inds = random.sample(mut_inds, n_nodes)
    mut_inds = random.sample(mut_inds, n_nodes)

    for i in mut_inds:
        mut_func = partial(mut_ind, i)
        gp_tree = perform_hillclimbing(toolbox, gp_tree, mut_func, hc_repeat=hc_repeat, keep_last=keep_last)

    return gp_tree


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

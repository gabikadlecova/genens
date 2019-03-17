# -*- coding: utf-8 -*-

import genens.config.clf_default as cf
from genens.gp import operators, types

from copy import deepcopy
from deap import base, creator

if __name__ == "__main__":
    creator.create("GenensFitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("TreeIndividual", types.GpTreeIndividual, fitness=creator.GenensFitness)
    config = cf.create_config(100)

    tree1 = operators.gen_tree(config, 7)
    tree2 = operators.gen_tree(config, 7)

    tree1_copy = deepcopy(tree1)
    tree2_copy = deepcopy(tree2)

    assert tree1 == tree1
    assert tree1 != tree2

    assert tree1 == tree1_copy
    assert tree2 == tree2_copy

    assert tree1 is not tree1_copy
    assert tree2 is not tree2_copy

# -*- coding: utf-8 -*-
"""
Module for visualization of the trees.
"""

import pygraphviz as pgv

from genens.gp.types import GpTreeIndividual, GpPrimitive


def create_graph(gp_tree, file_name):
    graph = pgv.AGraph()
    i = 0

    def add_node(node, child_list):
        nonlocal i
        node_id = i
        i = i + 1

        graph.add_node(node_id, label=node.name)

        for val in child_list:
            graph.add_edge(node_id, val)

        return node_id

    gp_tree.run_tree(add_node)

    graph.layout(prog='dot')
    graph.draw(file_name, prog='dot')
    graph.close()


def tree_str(gp_tree: GpTreeIndividual, with_hyperparams=False):

    def get_tree(node, ch_l):
        return node, ch_l

    def print_node(node: GpPrimitive, ch_l, indent, str_res):
        str_res += "  " * indent + node.name + "\n"
        if with_hyperparams:
            for k, val in node.obj_kwargs.items():
                str_res += "  " * (indent + 1) + "| " + f"{k}: {val}"

        for child, next_list in ch_l:
            str_res = print_node(child, next_list, indent + 1, str_res)

        return str_res

    root, child_list = gp_tree.run_tree(get_tree)
    return print_node(root, child_list, 0, "")

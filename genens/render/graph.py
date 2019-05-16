# -*- coding: utf-8 -*-
"""
Module for visualization of the trees.
"""

import pygraphviz as pgv


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


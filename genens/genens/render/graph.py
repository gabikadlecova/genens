# -*- coding: utf-8 -*-

import pygraphviz as pgv


def create_graph(gp_tree, file_name):
    graph = pgv.AGraph()

    def add_node(node, child_list):
        graph.add_node(node.name)

        for val in child_list:
            graph.add_edge(node.name, child_list)

        return node.name

    gp_tree.run_tree(add_node)
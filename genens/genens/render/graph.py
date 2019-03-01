# -*- coding: utf-8 -*-

import pygraphviz as pgv


def create_graph(gp_tree, fname): 
    
    class StackElem:
        def __init__(self, name, id, arity_rem):
            self.name = name
            self.id = id
            self.arity_rem = arity_rem
    
    graph = pgv.AGraph()
    
    stack = []
    
    # TODO height only 1
    for i, node in enumerate(gp_tree.primitives):
        graph.add_node(i, label=node.name)
        
        if node.arity > 0:
            stack.append(StackElem(node.name, i, node.arity))
        else:
            curr_id = i
            
            while len(stack):
                top = stack[-1]
                graph.add_edge(top.id, curr_id)
                
                top.arity_rem = top.arity_rem - 1
                if top.arity_rem == 0:
                    curr_id = top.id
                    stack.pop()
                else:
                    break
                
    graph.layout(prog='dot')
    graph.draw(fname, prog='dot')


def create_graph2(gp_tree, file_name):
    graph = pgv.AGraph()

    def add_node(node, child_list):
        graph.add_node(node.name)

        for val in child_list:
            graph.add_edge(node.name, child_list)

        return node.name

    gp_tree.run_tree(add_node)
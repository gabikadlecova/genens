# -*- coding: utf-8 -*-

"""This file defines the structure of GP primitives.
The GP primitives used in our project are variable-arity nodes with input and
output transformations.

The primitives defined in this file are
    1) functions - inner nodes of the tree, transform input into output
    2) terminals - leaves of the tree, provide constant output.
"""

# TODO check docs if it's still valid

import random
import functools
from abc import ABC, abstractmethod


class GpTreeIndividual:
    """Represents a tree individual used in the GP.
    The individual is a tree encoded in a list of ``GpPrimitive`` nodes.

    The list represents a pre-order traversal of the tree. The tree can be
    uniquely reconstructed by using the arity of primitives.
    """
    # TODO validate arity etc
    def __init__(self, prim_list, tree_id):
        self.primitives = prim_list
        self.id = tree_id

    def run_tree(self, node_func):
        stack = []

        # TODO height only 1
        for node in reversed(self.primitives):
            if node.arity == 0:
                stack.append(node_func(node, []))
            else:
                args = stack[-node.arity:]
                stack = stack[:-node.arity]

                stack.append(node_func(node, args))

        # TODO error

        return stack.pop()



class GpPrimitive(ABC):
    """Represents a primitive in the GP tree.
    This is the minimal definition of a primitive which can be used in the GP
    tree.
    """

    # TODO check validity
    def __init__(self, name, obj, node_type, arity):
        """Sets up the GP primitive.

        Args:
            TODO missing docstring
            node_type : tuple
                A tuple of format ``(inputs, output)``, where ``inputs`` is
                a list of input types of length ``arity`` and ``output`` is
                the type of the output.
            arity : int
                Arity of ``in_func``, which also means it is the arity of
                the corresponding tree node.
        """
        self.name = name
        self.obj = obj
        self.node_type = node_type
        self.arity = arity

    class PrimType:
        def __init__(self, name, arity):
            self.name = name
            self.arity = arity


class GpFunction(GpPrimitive):
    """Represents an inner node of the GP tree.
    This object encapsulates a function, which transforms inputs
    to output. It is an inner node of the GP tree.

    Its ``arity`` is greater than 0 and ``node_type`` is
    (inputs, output).
    """
    def __init__(self, name, obj, node_type, arity):
        # TODO check arity
        super().__init__(name, obj, node_type, arity)


class GpTerminal(GpPrimitive):
    """Represents a leaf of the GP tree.
    This object encapsulates a constant, which provides output to a parent
    inner node.

    Its ``arity`` is set to 0 and ``node_type`` is (, output).
    """
    def __init__(self, name, obj, node_type):
        # TODO validate type
        super().__init__(name, obj, node_type, 0)


class TypeArity:
    """Represents variable arity.
    """
    def __init__(self, prim_type, arity_range):
        """Arity range (from, to), to can be 'n' (up to current maximal arity)
        """
        self.prim_type = prim_type
        self.arity_range = arity_range
        
    def choose_arity(self, max_arity):
        # TODO validate the range
        a_from, a_to = self.arity_range
        
        if a_to == 'n' or a_to > max_arity:
            a_to = max_arity
            
        return random.randint(a_from, a_to)
        

class FunctionTemplate:
    """Represents a template of a GpPrimitive with variable arities of inputs.
    
    TODO explain (type arities is sth like (type, max arity))
    """
    def __init__(self, name, obj, type_arities, out_type):
        self.name = name
        self.obj = obj
        self.type_arities = type_arities
        self.out_type = out_type
        
    def create_primitive(self, max_arity):
        """Creates a GpPrimitive from the template.
        Creates a primitive according to the template.
        
        The ``in_func`` and ``out_func`` parameters remain the same.
        Type and arity is created from the list of possible arities per type.
        
        TODO example of type creation
        TODO describe arities
        """
        def create_type(t):
            arity = t.choose_arity(max_arity)
            return GpPrimitive.PrimType(t.prim_type, arity)
        
        in_type = [create_type(t_a) for t_a in self.type_arities]
        arity_sum = functools.reduce(lambda s, t: s + t.arity, in_type, 0)
        
        return GpFunction(self.name, self.obj, (in_type, self.out_type), arity_sum)

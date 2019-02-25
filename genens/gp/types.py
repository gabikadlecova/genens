# -*- coding: utf-8 -*-

"""This file defines the structure of GP primitives.
The GP primitives used in our project are variable-arity nodes with input and
output transformations.

The primitives defined in this file are
    1) functions - inner nodes of the tree, transform input into output
    2) terminals - leaves of the tree, provide constant output.
"""

from abc import ABC, abstractmethod

class GpTreeIndividual:
    """Represents a tree individual used in the GP.
    """
    pass


class GpPrimitive(ABC):
    """Represents a primitive in the GP tree.
    This is the minimal definition of a primitive which can be used in the GP
    tree.
    """

    # TODO check validity
    def __init__(self, in_func, out_func, node_type, arity):
        """Sets up the GP primitive.

        Args:
            in_func : callable
                Function that takes ``arity`` inputs and transforms them.
            out_func : callable
                Function that takes output of ``in_func`` as its argument(s)
                and transforms it to provide final output.
            node_type : tuple
                A tuple of format ``(inputs, output)``, where ``inputs`` is
                a list of input types of length ``arity`` and ``output`` is
                the type of the output.
            arity : int
                Arity of ``in_func``, which also means it is the arity of
                the corresponding tree node.
        """
        self.in_func = in_func
        self.out_func = out_func
        self.node_type = node_type
        self.arity = arity

    @abstractmethod
    def run_primitive(self, inputs):
        """Transforms inputs to produce node output.
        """


class GpFunction(GpPrimitive):
    """Represents an inner node of the GP tree.
    This object encapsulates a function, which transforms inputs
    to output. It is an inner node of the GP tree.

    Its ``arity`` is greater than 0 and ``node_type`` is
    (inputs, output).
    """
    def __init__(self, in_func, out_func, node_type, arity):
        # TODO check arity
        super().__init__(in_func, out_func, node_type, arity)

    def run_primitive(self, inputs):
        # TODO handle possible errors
        transf_in = self.in_func(inputs)
        return self.out_func(transf_in)


class GpTerminal(GpPrimitive):
    """Represents a leaf of the GP tree.
    This object encapsulates a constant, which provides output to a parent
    inner node.

    Its ``arity`` is set to 0 and ``node_type`` is (, output).
    """
    def __init__(self, out_func, node_type):
        # TODO validate type
        super().__init__(None, out_func, node_type, 0)
        
    def run_primitive(self, inputs):
        # TODO handle possible errors
        return self.out_func(inputs)

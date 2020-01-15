# -*- coding: utf-8 -*-

"""
This module defines the structure of GP primitives.

The GP primitives are nodes with typed edges (parent input and child output types must match) and variable
arity (for a given type, its final arity can be chosen during the evolution process).

A ``GpPrimitive`` is a node whose types, arities and keyword arguments have been decided. To create such primitives,
it is possible to take use of templates. These contain possible values of arities, types and keyword arguments and
methods for choosing final values.

The primitive templates defined in this file are
    1) functions - inner nodes of the tree, transform input into output
    2) terminals - leaves of the tree, provide constant output.
"""


import random
import functools
from copy import deepcopy

from deap import base


class GpTreeIndividual:
    """Represents a tree individual used in the GP.
    The individual is a tree encoded as a list of ``GpPrimitive`` nodes.

    The list is a post-order representation of the tree. The tree can be
    uniquely reconstructed using the arity (and types) of primitives.
    """
    def __init__(self, prim_list, max_height):
        """
        Construct a GP tree from a list of primitives.

        :param list of GpPrimitive prim_list: Post-order representation of the tree.
        """
        self.primitives = prim_list
        self.max_height = max_height

        self.validate_tree()

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        new = object.__new__(type(self))
        memo[id(self)] = new
        new.__dict__.update(deepcopy(self.__dict__, memo))

        return new

    def __eq__(self, other):
        if not isinstance(other, GpTreeIndividual):
            return False

        if self.primitives != other.primitives:
            return False

        return True

    def __repr__(self):
        return 'GpTreePrimitive(height={}, '.format(self.max_height)\
               + 'primitives=' + self.primitives.__repr__() + ')'

    def run_tree(self, node_func):
        """
        Applies a function with the signature ``func(node, child_list)`` on all
        nodes in the tree. The tree is traversed in post-order.

        The arguments of the function are a node and list of result values of its child nodes.

        :param node_func: Function which is applied on all nodes of the tree.
        :return: Return value of the root.
        """
        stack = []

        for node in self.primitives:
            if node.arity == 0:
                stack.append(node_func(node, []))
            else:
                args = stack[-node.arity:]
                stack = stack[:-node.arity]

                stack.append(node_func(node, args))

        if len(stack) > 1:
            raise ValueError("Bad tree")  # TODO specific

        return stack.pop()

    def subtree(self, root_ind):
        """
        Returns the end position of the subtree in the tree individual.

        :param root_ind: Position of the root (index of the beginning).
        :return: The end index of the subtree.
        """
        curr = self.primitives[root_ind]

        arity_rem = curr.arity
        init_h = curr.depth
        max_h = init_h

        while arity_rem > 0:
            root_ind = root_ind - 1
            curr = self.primitives[root_ind]
            max_h = max(max_h, curr.depth)

            arity_rem = arity_rem - 1 + curr.arity

        return root_ind, (max_h - init_h + 1)

    def validate_tree(self):
        """
        Validates the tree, raises an exception if its children are invalid.
        """

        def validate_node(node, child_list):
            if node.arity != len(child_list):
                raise ValueError("Invalid number of children.")  # TODO specific

            child_id = 0
            for in_type in node.node_type[0]:
                for i in range(in_type.arity):
                    child = child_list[child_id + i]
                    if child.node_type[1] != in_type.name:
                        raise ValueError("Invalid child type.")  # TODO specific

                    if child.depth != node.depth + 1:
                        raise ValueError("Invalid child height.")  # TODO specific

                child_id += in_type.arity

            return node

        self.run_tree(validate_node)

        if self.max_height != max(prim.depth + 1 for prim in self.primitives):
            raise ValueError("Invalid tree height.")  # TODO specific


class DeapTreeIndividual(GpTreeIndividual):
    """
    Represents an individual which contains DEAP defined fitness.
    """
    def __init__(self, prim_list, max_height):
        super().__init__(prim_list, max_height)
        self.fitness = DeapTreeIndividual.Fitness()
        self.compiled_pipe = None
        self.test_stats = None

    class Fitness(base.Fitness):
        def __init__(self, values=()):
            self.weights = (1.0, -1.0)
            super().__init__(values)

    def reset(self):
        del self.fitness.values

        self.compiled_pipe = None
        self.test_stats = None


class GpPrimitive:
    """
    Represents a node in the GP tree. It is defined by its type and
    arity values.

    Its name and keyword dictionary hold information about the function
    or object, which is associated with the node.
    """

    def __init__(self, name, obj_kwargs, node_type, arity, depth):
        """
        Creates an instance of a GP tree node. Number of its children
        is specified by its arity, types of children are determined by its
        PrimType type. Its children must be ordered according to the input
        type list.

        :param str name: Name key associated with the node.
        :param dict obj_kwargs: Keyword arguments associated with the node.
        :param (list of PrimType, str) node_type:
            A tuple, where first item is a list of input types with arities
            and the second item is output type name.
        :param int arity: Sum of all arities of types.
        :param int height: Depth of the node.
        """
        self.name = name
        self.obj_kwargs = obj_kwargs
        self.node_type = node_type
        self.arity = arity
        self.depth = depth

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        new = object.__new__(type(self))
        memo[id(self)] = new
        new.__dict__.update(deepcopy(self.__dict__, memo))

        return new

    def __eq__(self, other):
        if not isinstance(other, GpPrimitive):
            return False

        if self.name != other.name:
            return False

        if self.arity != other.arity or self.node_type != other.node_type:
            return False

        if self.obj_kwargs != other.obj_kwargs:
            return False

        return True

    def __repr__(self):
        return 'GpPrimitive(name=' + self.name + ", arity={}".format(self.arity)\
               + ", height={})".format(self.depth)

    @property
    def out_type(self):
        return self.node_type[1]

    class PrimType:
        """
        Represents the input type which must match the output type of
        a certain number of children (determined by ``arity``).
        """
        def __init__(self, name, arity):
            """
            Construct a new instance of the type associated with an arity.

            :param name: Name of the type.
            :param arity: Arity associated with this type.
            """
            self.name = name
            self.arity = arity

        def __eq__(self, other):
            if not isinstance(other, GpPrimitive.PrimType):
                return False

            if self.name != other.name or self.arity != other.arity:
                return False

            return True


class GpTerminalTemplate:
    """
    Represents a leaf of the GP tree.
    Creates a primitive with no inputs. The output type is specified.

    The ``arity`` of the primitive will be set to 0 and ``node_type`` is set to ``([], out_type)``.

    The keyword arguments are chosen from lists of possible values.
    """
    def __init__(self, name, out_type, group=None):
        """
        Creates a new instance of a terminal template.

        :param str name: Name key associated with the node.
        :param str out_type: Name of the node output type.
        """
        self.name = name
        self.type_arities = []
        self.out_type = out_type
        self.group = group

    def create_primitive(self, curr_height, max_arity, kwargs_dict):
        """
        Creates an instance of a ``GpPrimitive`` from the template.

        Selects keyword arguments from ``kwargs_dict``. For every key,
        the dict should contain a list of possible values.

        :param int curr_height: Height at which the node is generated.
        :param int max_arity: Maximum arity of all nodes in the tree, not used.
        :param dict kwargs_dict: Dictionary which contains possible keyword argument values.
        :return: A new instance of GpPrimitive.
        """
        prim_kwargs = _choose_kwargs(kwargs_dict)

        prim_type = ([], self.out_type)

        return GpPrimitive(self.name, prim_kwargs, prim_type, 0, curr_height)


class TypeArity:
    """
    Represents a variable node arity associated with a type.
    """
    def __init__(self, prim_type, arity_range):
        """
        Constructs a new instance of the typed arity.

        :param str prim_type: Type associated with the arity.
        :param int or (int, int) or (int, 'n') arity_range:
            Arity of input nodes, either a constant, or a range (inclusive), or a range
            without a specified upper bound (in the case of (int, 'n')).
        """
        self.prim_type = prim_type
        self.arity_range = arity_range

        if isinstance(self.arity_range, tuple):
            if self.arity_range[1] != 'n' and self.arity_range[0] > self.arity_range[1]:
                raise ValueError("Invalid arity range.")  # TODO specific
        elif isinstance(self.arity_range, int):
            if self.arity_range <= 0:
                raise ValueError("Arity must be greater than 0.")  # TODO specific
        else:
            raise ValueError("Invalid arity type.")  # TODO specific

    def is_valid_arity(self, arity):
        if isinstance(self.arity_range, tuple):
            if arity < self.arity_range[0]:
                return False

            if self.arity_range[1] != 'n' and arity > self.arity_range[1]:
                return False

            return True

        if isinstance(self.arity_range, int):
            return self.arity_range == arity

        raise ValueError("Invalid arity object.")  # TODO specific

    def choose_arity(self, max_arity):
        """
        Chooses an integer arity from the arity range or
        returns the arity in case of a single int.

        :param int max_arity:
            The upper bound of arities. Is applied for variable upper bounds and for
            ranges where the lower bound is less than this parameter.
        :return int: Returns a number from the range of possible arities.
        """
        if not isinstance(self.arity_range, tuple):
            return self.arity_range

        a_from, a_to = self.arity_range

        # is not applied for ranges which are greater than ``max_arity`` as it could break them
        if a_to == 'n' or a_to > max_arity >= a_from:
            a_to = max_arity
            
        return random.randint(a_from, a_to)
        

class GpFunctionTemplate:
    """
    Represents an inner node of the GP tree.
    Creates a primitive which accepts inputs. The output type is also specified.

    The ``arity`` of the primitive is determined by arities associated with types. These
    are ordered (the node children must maintain the same order from left to right). The
    final arities are chosen when a primitive is created from the template.

    The keyword arguments are chosen from lists of possible values.
    """
    def __init__(self, name, type_arities, out_type, group=None):
        """
        Creates a new instance of a function template.

        :param str name: Name key associated with the node.
        :param list of TypeArity type_arities:
            Ordered list of children arity ranges (associated with a type).
        :param str out_type: Name of the output type.
        """
        self.name = name
        self.type_arities = type_arities
        self.out_type = out_type
        self.group = group
        
    def create_primitive(self, curr_height, max_arity, kwargs_dict, in_type=None):
        """
        Creates an instance of a ``GpPrimitive`` from the template.

        Selects keyword arguments from ``kwargs_dict``. For every key,
        the dict should contain a list of possible values.

        Select final arities of every TypeArity in the list. If some of
        the arities result in 0, they are not added into the final input type.

        :param int curr_height: Height at which the node is generated
        :param int max_arity: Maximum arity value which can be chosen for a single TypeArity.
        :param dict kwargs_dict: Dictionary which contains possible keyword argument values.
        :param in_type: Input type; if provided, it is used instead of generating a new random type.

        :return: A new instance of GpPrimitive
        """
        prim_kwargs = _choose_kwargs(kwargs_dict)

        def create_type(t):
            arity = t.choose_arity(max_arity)

            # do not construct a type in case of 0
            if arity == 0:
                return None

            return GpPrimitive.PrimType(t.prim_type, arity)

        if in_type is None:
            # ordered list of final typed arities
            in_type = [create_type(t_a) for t_a in self.type_arities]
            in_type = [t for t in in_type if t is not None]

        # TODO check the provided in_type (and maybe change it a bit)

        # sum of all arities
        arity_sum = functools.reduce(lambda s, t: s + t.arity, in_type, 0)
        
        return GpPrimitive(self.name, prim_kwargs, (in_type, self.out_type), arity_sum, curr_height)


def _choose_kwargs(kwargs_dict):
    """
    Chooses keyword argument values from the argument dictionary, which contains
    list of possible values for every key.

    :param dict kwargs_dict: Dictionary of possible key values.
    :return dict: Dictionary with one value selected per key.
    """
    if kwargs_dict is None:
        return {}

    chosen_kwargs = {}

    for keyword, arg_list in kwargs_dict.items():
        arg = random.choice(arg_list)
        chosen_kwargs[keyword] = arg

    return chosen_kwargs

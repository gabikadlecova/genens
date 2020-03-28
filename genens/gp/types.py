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


import functools
import random

from copy import deepcopy
from deap import base
from typing import Callable, List, Any, Dict, Union, Tuple


class GpTreeIndividual:
    """Represents a tree individual used in GP.
    The individual is a tree encoded as a list of ``GpPrimitive`` nodes.

    The list is a post-order representation of the tree. The tree can be
    uniquely reconstructed using the arity (and types) of primitives.
    """
    def __init__(self, prim_list: List['GpPrimitive'], max_height: int):
        """
        Construct a GP tree from a list of primitives.

        Args:
            prim_list: list of GpPrimitive prim_list: Post-order representation of the tree.
            max_height: Height of the tree - maximum of all node depths + 1
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
        return f'GpTreeIndividual(height={self.max_height} primitives={self.primitives.__repr__()})'

    def run_tree(self, node_func, group_children: bool = False) -> Any:
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

                if group_children:
                    children_by_type = []
                    for t in node.in_type:
                        t_children, args = args[-t.arity:], args[:-t.arity]
                        children_by_type.append((t.name, t_children))

                    args = children_by_type

                stack.append(node_func(node, args))

        if len(stack) > 1:
            raise ValueError("Bad tree - invalid primitive list.")

        return stack.pop()

    def subtree(self, root_ind: int) -> Tuple[int, int]:
        """
        Returns the start position of the subtree with primitive `self.primitives[root_ind]` as root. Note
        that the returned value is in fact the index of one of the leaves, as the node list is post-order.
        As so, the whole subtree is extracted with `self.primitives[subtree(root_ind), root_ind + 1]`.

        Args:
            root_ind: Position of the root (index of the beginning).

        Returns: A tuple `(s, h)`, where `s` is the start index and `h` is the height of the subtree.
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
                raise ValueError("Invalid number of children.")

            child_id = 0
            for in_type in node.in_type:
                for i in range(in_type.arity):
                    child = child_list[child_id + i]
                    if child.out_type != in_type.name:
                        raise ValueError("Invalid child type.")

                    if child.depth != node.depth + 1:
                        raise ValueError("Invalid child height.")

                child_id += in_type.arity

            return node

        self.run_tree(validate_node)

        if self.max_height != max(prim.depth + 1 for prim in self.primitives):
            raise ValueError("Invalid tree height.")


class DeapTreeIndividual(GpTreeIndividual):
    """
    Represents an individual with DEAP-compatible fitness.
    """
    def __init__(self, prim_list: List['GpPrimitive'], max_height: int):
        super().__init__(prim_list, max_height)

        self.fitness = DeapTreeIndividual.Fitness()  # (score, log(evaluation_time))
        self.compiled_pipe = None

    class Fitness(base.Fitness):
        def __init__(self, values=()):
            self.weights = (1.0, -1.0)
            super().__init__(values)

    def reset(self):
        del self.fitness.values

        self.compiled_pipe = None


class GpPrimitive:
    """
    Represents a typed node in the GP tree.

    Its name and keyword dictionary hold information about the function
    or object, which is associated with the node.
    """

    def __init__(self,
                 name: str,
                 obj_kwargs: Dict[str, Any],
                 in_type: List['GpPrimitive.InputType'],
                 out_type: str,
                 arity: int,
                 depth: int):
        """
        Creates an instance of a GP tree node. The number and output types as well as the ordering of its children
        is specified by `in_type`.

        Args:
            name: Name of the node.
            obj_kwargs: Keyword arguments associated with the node.
            in_type:
                List of input types with arity. The subtypes are ordered - e.g. [('data', 2), ('ens', 1)] is not the
                same as [('ens', 1), ('data', 2)].
            arity: Sum of arity of subtypes.
            depth: Depth of the node.
        """
        self.name = name
        self.obj_kwargs = obj_kwargs
        self.in_type = in_type
        self.out_type = out_type
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

        if self.arity != other.arity or self.in_type != other.in_type or self.out_type != other.out_type:
            return False

        if self.obj_kwargs != other.obj_kwargs:
            return False

        return True

    def __repr__(self):
        return 'GpPrimitive(name=' + self.name + ", arity={}".format(self.arity)\
               + ", height={})".format(self.depth)

    class InputType:
        """
        Represents the input type of a primitive. It determines how many children with a specific output type
        should the node have.
        """
        def __init__(self, name: str, arity: int):
            """
            Construct a new instance of input type with arity.

            :param name: Name of the type.
            :param arity: Arity of this type.
            """
            self.name = name
            self.arity = arity

        def __eq__(self, other):
            if not isinstance(other, GpPrimitive.InputType):
                return False

            if self.name != other.name or self.arity != other.arity:
                return False

            return True


class GpTerminalTemplate:
    """
    Represents a terminal of the GP tree, or a primitive with no inputs.
    The output type is fixed.

    The keyword arguments are chosen from lists of possible values.
    """
    def __init__(self, name: str, out_type: str, group: str = None):
        """
        Creates a new instance of a terminal template.

        Args:
            name: Name of the node.
            out_type: Name of the output type.
        """
        self.name = name
        self.type_arity_template = []
        self.out_type = out_type
        self.group = group

    def create_primitive(self, curr_height: int, max_arity: int, kwargs_dict: Dict[str, List[Any]]) -> GpPrimitive:
        """
        Creates an instance of a `GpPrimitive` from the template.

        Selects keyword arguments from `kwargs_dict`. For every key,
        the dict contains a list of possible values.

        Args:
            curr_height: Height at which the node is generated.
            max_arity: Only for compatibility, not used for terminals.
            kwargs_dict: Dictionary which contains possible keyword argument values.

        Return: A new instance of `GpPrimitive`.
        """
        prim_kwargs = _choose_kwargs(kwargs_dict)

        return GpPrimitive(self.name, prim_kwargs, [], self.out_type, 0, curr_height)


class TypeArity:
    """
    Represents a variable node arity associated with a type.
    """
    def __init__(self,
                 prim_type: str,
                 arity_template: Union[int, Tuple[int, int], Tuple[int, str]]):
        """
        Constructs a new instance of a type template - with a fixed type, but possibly variable arity.
        Args:
            prim_type: Name of the type.
            arity_template: Either a fixed arity value, or a bounded interval (a, b) where a and b are integers,
                or an interval (a, 'n') that has only a lower bound a ('n' is a string).
        """
        self.prim_type = prim_type
        self.arity_template = arity_template

        # check arity range
        if isinstance(self.arity_template, tuple):
            lower_invalid = self.arity_template[0] < 0
            upper_invalid = self.arity_template[1] != 'n' and self.arity_template[0] > self.arity_template[1]

            if lower_invalid or upper_invalid:
                raise ValueError("Invalid arity range.")

        # check fixed arity
        elif isinstance(self.arity_template, int):
            if self.arity_template <= 0:
                raise ValueError("Arity must be greater than 0.")
        else:
            raise ValueError("Invalid arity type.")

    def is_valid_arity(self, arity: int):
        """
        Determines whether `self.choose_arity` could possibly result in `arity`.
        Args:
            arity: Input arity to compare with this template.

        Returns: True if `arity` can be created from this template.

        """
        if isinstance(self.arity_template, tuple):
            # out of range
            if arity < self.arity_template[0]:
                return False

            if self.arity_template[1] != 'n' and arity > self.arity_template[1]:
                return False

            # inside range
            return True

        # match fixed arity
        if isinstance(self.arity_template, int):
            return self.arity_template == arity

        return False

    def choose_arity(self, max_arity: int) -> int:
        """
        Chooses an integer arity from the arity range or
        returns the arity if it is already a fixed value.

        Args:
            max_arity: The upper bound of arities. It is an upper bound for all ranges, even when the arity
                upper bound is greater. If the lower bound is greater than this value, `max_arity` is not applied
                and the value is chosen from the original interval.

        Returns: A fixed arity value.
        """
        if not isinstance(self.arity_template, tuple):
            return self.arity_template

        a_from, a_to = self.arity_template

        # is not applied for ranges which are greater than ``max_arity`` as it could result in invalid behavior
        if a_to == 'n' or a_to > max_arity >= a_from:
            a_to = max(max_arity, a_from)

        return random.randint(a_from, a_to)
        

class GpFunctionTemplate:
    """
    Represents an inner node of the GP tree.
    This class is a template of a function node - the input type may have variable arity which is decided when
    an instance of `GpPrimitive` is created from this template. During this process, keyword arguments are decided
    as well.

    The template input type is an ordered list of subtypes. These may have a variable arity (a range) and the
    final arity is chosen during the instantiation.

    The keyword arguments are chosen from lists of possible values.
    """
    def __init__(self, name: str, type_arity_template: List[TypeArity], out_type: str, group: str = None):
        """

        Args:
            name: Name key associated with the node.
            type_arity_template: Ordered list of children subtypes with variable arity.
            out_type: Name of the output type.
            group: Name of group of the node.
        """

        self.name = name
        self.type_arity_template = type_arity_template
        self.out_type = out_type
        self.group = group
        
    def create_primitive(self,
                         curr_height: int,
                         max_arity: int,
                         kwargs_dict: Dict[str, List],
                         in_type: GpPrimitive.InputType = None) -> GpPrimitive:
        """
        Creates an instance of a ``GpPrimitive`` from the template.

        Selects keyword arguments from ``kwargs_dict``. For every key,
        the dict contains a list of possible values.

        Select final arities of every TypeArity in `self.type_arities` - that is, if a function node has
        a variable number of input arguments (possibly of different types), for every subtype a fixed arity
        is chosen.

        TODO add example

        Args:
            curr_height: Height at which the node is generated
            max_arity: Maximum arity value which can be chosen for a single TypeArity.
            kwargs_dict: Dictionary which contains possible keyword argument values.
            in_type: Input type; if provided, it is used instead of generating a new random type.

        Returns: A new instance of GpPrimitive
        """
        prim_kwargs = _choose_kwargs(kwargs_dict)

        def create_type(t):
            arity = t.choose_arity(max_arity)

            # do not construct a type in case of 0
            if arity == 0:
                return None

            return GpPrimitive.InputType(t.prim_type, arity)

        if in_type is None:
            # ordered list of final arity of types
            in_type = [create_type(t_a) for t_a in self.type_arity_template]
            in_type = [t for t in in_type if t is not None]

        #
        # total arity of the node
        arity_sum = functools.reduce(lambda s, t: s + t.arity, in_type, 0)
        
        return GpPrimitive(self.name, prim_kwargs, in_type, self.out_type, arity_sum, curr_height)


def _choose_kwargs(kwargs_dict: Dict[str, List]) -> Dict[str, Any]:
    """
    Chooses keyword argument values from the argument dictionary `kwargs_dict`.
    For every keyword argument, it contains a list of possible values for every key.

    Args:
        kwargs_dict: Dict of possible kwarg values.

    Returns: Dict with one value selected per key.

    """
    if kwargs_dict is None:
        return {}

    return {k: random.choice(arg_list) for k, arg_list in kwargs_dict.items()}

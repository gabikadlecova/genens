import json
import warnings
import yaml

from functools import partial
from genens.config.utils import import_custom_func
from genens.gp.types import TypeArity, GpTerminalTemplate, GpFunctionTemplate

from typing import Union, Dict


class GenensConfig:
    """
    Configuration of Genens estimators. Contains settings of GP algorithm and individuals
    and configuration of methods that decode nodes into scikit-learn models.
    """
    def __init__(self, function_config=None, full_config=None, term_config=None, kwargs_config=None,
                 cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.9, mut_node_pb=0.9,
                 hc_repeat=0, hc_keep_last=False, hc_mut_pb=0.2, hc_n_nodes=3,
                 mut_multiple_args=False, mut_multiple_nodes=False,
                 min_height=1, max_height=4, min_arity=2, max_arity=3, group_weights=None):
        """
        Creates a instance of genens configuration.

        Args:
            function_config: Function configuration dictionary, keys correspond do a particular primitive name.
            full_config: Node configuration dictionary, nodes are grouped by output types,
                contains node templates used during the grow phase.

            term_config: Node configuration dictionary, nodes are grouped by output types,
                contains node templates used to terminate a tree.

            kwargs_config: Keyword arguments for nodes, which are passed to functions during the decoding.
                For every argument, there is a list of possibles values to choose from during the evolution.

            min_height: Minimum height of trees.
            max_height: Maximum height of trees.
            min_arity: Lower arity limit on any subtype.
            max_arity: Upper arity limit on any subtype. A node may have a larger total arity than this
                limit, but none of its subtypes can exceed it.
                E.g. for `max_arity` == 2 the node type (out^2, data^1) is valid - the arity of the node
                is 3, but its subtypes do not exceed the limit.

            group_weights: Weights of node groups, used during node selection. Nodes with a higher weight
                have a greater probability to appear in trees.
        """
        self.func_config = function_config if function_config is not None else {}
        self.full_config = full_config if full_config is not None else {}
        self.term_config = term_config if term_config is not None else {}
        self.kwargs_config = kwargs_config if kwargs_config is not None else {}

        self.min_height = min_height
        self.max_height = max_height
        self.min_arity = min_arity
        self.max_arity = max_arity

        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.mut_args_pb = mut_args_pb
        self.mut_node_pb = mut_node_pb

        self.hc_repeat = hc_repeat
        self.hc_keep_last = hc_keep_last
        self.hc_mut_pb = hc_mut_pb
        self.hc_n_nodes = hc_n_nodes

        self.mut_multiple_nodes = mut_multiple_nodes
        self.mut_multiple_args = mut_multiple_args

        self.group_weights = group_weights if group_weights is not None else {}

    def add_terminal(self, prim: GpTerminalTemplate, leaf_only: bool = False):
        """
        Adds a new primitive to the configuration - to both full and term dictionaries.
        If `term_only` is True, it is added only to the terminal set.

        Args:
            prim: Primitive to be added.
            leaf_only: Specifies whether the primitive should be added only to the terminal set.
        """

        if not leaf_only:
            out_list = self.full_config.setdefault(prim.out_type, [])
            out_list.append(prim)

        out_list = self.term_config.setdefault(prim.out_type, [])
        out_list.append(prim)

    def add_function(self, prim):
        """
        Add a function to the configuration - to the full set.

        Args:
            prim: Primitive to be added.
        """
        out_list = self.full_config.setdefault(prim.out_type, [])
        out_list.append(prim)


def parse_config(config_path: str, base_config: GenensConfig = None, evo_kwargs: dict = None):
    config = base_config if base_config is not None else GenensConfig()

    with open(config_path, 'r') as input_file:
        yaml_config = yaml.safe_load(input_file)

    min_height = yaml_config.get('min_height', config.min_height)
    max_height = yaml_config.get('max_height', config.max_height)
    min_arity = yaml_config.get('min_arity', config.min_arity)
    max_arity = yaml_config.get('max_arity', config.max_arity)

    group_weights = yaml_config.get('group_weights', {})

    load_kwargs_from_yaml = evo_kwargs is None
    evo_kwargs = {} if evo_kwargs is None else evo_kwargs

    primitives = yaml_config.get('primitives', None)
    func_dict, full_dict, term_dict, evo_kwargs_dict = parse_primitives(primitives,
                                                                        parse_evo_kwargs=load_kwargs_from_yaml)

    return GenensConfig(function_config={**config.func_config, **func_dict},
                        full_config={**config.full_config, **full_dict},
                        term_config={**config.term_config, **term_dict},
                        kwargs_config={**config.kwargs_config, **evo_kwargs_dict, **evo_kwargs},
                        min_height=min_height,
                        max_height=max_height,
                        min_arity=min_arity,
                        max_arity=max_arity,
                        group_weights={**config.group_weights, **group_weights})


def parse_primitives(primitives, parse_evo_kwargs=True):
    func_dict = {}
    full_dict = {}
    term_dict = {}

    evo_kwargs_dict = {}

    for prim_key, values in primitives.items():
        parsed_prim = parse_primitive(prim_key, values)

        # optionally parse kwargs that are changed during evolution
        if 'evo_kwargs' in values:
            if not parse_evo_kwargs:
                warnings.warn("The parameter evo_kwargs is present yaml config, but it won't be loaded, "
                              "possibly because a dict with hyperparameters has been provided.")
            else:
                evo_kwargs_dict[prim_key] = values.get('evo_kwargs', {})

        prim = parsed_prim["prim"]
        # add to primitive sets
        if 'grow' in parsed_prim["set"]:
            _add_to_primset(prim, full_dict)

        if 'terminal' in parsed_prim["set"]:
            _add_to_primset(prim, term_dict)
            if "term_prim" in parsed_prim:
                _add_to_primset(parsed_prim["term_prim"], term_dict)

        func_dict[prim.name] = parsed_prim["func"]

    return func_dict, full_dict, term_dict, evo_kwargs_dict


def _add_to_primset(prim: Union[GpFunctionTemplate, GpTerminalTemplate], prim_dict: Dict):
    type_list = prim_dict.setdefault(prim.out_type, [])
    type_list.append(prim)


def parse_primitive(prim_name, prim_data):
    in_type = _parse_in_type(prim_data['in']) if 'in' in prim_data else None
    out_type = prim_data['out']

    func = _parse_func(prim_data['function'])
    group = prim_data.get('group', None)

    if in_type is None:
        prim = GpTerminalTemplate(prim_name, out_type, group=group)
    else:
        prim = GpFunctionTemplate(prim_name, in_type, out_type, group=group)

    parsed_prim = {
        'prim': prim,
        'func': func,
        'set': prim_data['set'].replace(' ', '').split(',')
    }

    if 'terminal_out' in prim_data:
        if 'terminal' not in parsed_prim['set']:
            warnings.warn('Terminal out type should be specified only for terminals, '
                          f'skipping for primitive {prim_name}')
        else:
            term_prim = GpTerminalTemplate(prim_name, prim_data['terminal_out'], group=group)
            parsed_prim['term_prim'] = term_prim

    return parsed_prim


def _parse_func(func_data: Union[str, Dict]):
    if isinstance(func_data, str):
        func_path = func_data
        func_kwargs = None
    else:
        func_path = func_data.pop('func')
        func_kwargs = func_data

    func = import_custom_func(func_path)
    return partial(func, **func_kwargs) if func_kwargs is not None else func


def _parse_in_type(in_type):
    result_type = []

    for subtype in in_type:
        type_name = subtype['name']
        if 'arity' in subtype:
            ta = TypeArity(type_name, subtype['arity'])
        elif 'from' in subtype and 'to' in subtype:
            ta = TypeArity(type_name, (subtype['from'], subtype['to']))
        else:
            raise ValueError("Invalid arity in config file.")

        result_type.append(ta)

    return result_type

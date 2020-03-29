import importlib
import json
import warnings
import yaml

from functools import partial
from genens.config.utils import GenensConfig
from genens.gp.types import TypeArity, GpTerminalTemplate, GpFunctionTemplate

from typing import Union, Dict


# TODO provide a base config
def parse_config(config_path: str, base_config: GenensConfig = None, evo_kwargs_json_path: str = None):
    config = base_config if base_config is not None else GenensConfig()

    with open(config_path, 'r') as input_file:
        yaml_config = yaml.safe_load(input_file)

    min_height = yaml_config.get('min_height', config.min_height)
    max_height = yaml_config.get('max_height', config.max_height)
    min_arity = yaml_config.get('min_arity', config.min_arity)
    max_arity = yaml_config.get('max_arity', config.max_arity)

    group_weights = yaml_config.get('group_weights', {})

    load_kwargs_from_yaml = evo_kwargs_json_path is None
    primitives = yaml_config.get('primitives', None)
    func_dict, full_dict, term_dict, evo_kwargs_dict = parse_primitives(primitives,
                                                                        parse_evo_kwargs=load_kwargs_from_yaml)

    if evo_kwargs_json_path is not None:
        with open(evo_kwargs_json_path, 'r') as json_file:
            json_evo_kwargs = json.load(json_file)
    else:
        json_evo_kwargs = {}

    return GenensConfig(function_config={**config.func_config, **func_dict},
                        full_config={**config.full_config, **full_dict},
                        term_config={**config.term_config, **term_dict},
                        kwargs_config={**config.kwargs_config, **evo_kwargs_dict, **json_evo_kwargs},
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

    for prim_key, values in primitives:
        prim, func, set = parse_primitive(prim_key, values)

        # optionally parse kwargs that are changed during evolution
        if 'evo_kwargs' in values:
            if not parse_evo_kwargs:
                warnings.warn("The parameter evo_kwargs is present yaml config, but it won't be loaded."
                              "Possibly there was a separate config file with evo_kwargs provided.")
            else:
                evo_kwargs_dict[prim_key] = values['evo_kwargs']

        # add to primitive sets
        if 'grow' in set:
            _add_to_primset(prim, full_dict)
        if 'terminal' in set:
            _add_to_primset(prim, term_dict)

    return func_dict, full_dict, term_dict, evo_kwargs_dict


def _add_to_primset(prim: Union[GpFunctionTemplate, GpTerminalTemplate], prim_dict: Dict):
    type_list = prim_dict.setdefault(prim.out_type, [])
    type_list.append(prim)


def parse_primitive(prim_name, prim_data):
    # TODO parse in type only for func
    in_type = _parse_in_type(prim_data['in']) if 'in' in prim_data else None
    out_type = prim_data['out']

    func = _parse_func(prim_data['function'])
    group = prim_data.get('group', None)

    if in_type is None:
        prim = GpTerminalTemplate(prim_name, out_type, group=group)
    else:
        prim = GpFunctionTemplate(prim_name, in_type, out_type, group=group)

    return prim, func, prim_data['set']


def _parse_func(func_data: Union[str, Dict]):
    if isinstance(func_data, str):
        func_path = func_data
        func_kwargs = None
    else:
        func_path = func_data.pop('func')
        func_kwargs = func_data

    func_path = func_path.split('.')
    func_name = func_path.pop()
    module_path = '.'.join(func_path)
    func = getattr(importlib.import_module(module_path), func_name)

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
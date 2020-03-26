import yaml
from genens.config.utils import GenensConfig


def parse_config(config_path):
    config = GenensConfig()

    with open(config_path, 'r') as input_file:
        yaml_config = yaml.safe_load(input_file)

    min_height = yaml_config.get('min_height', config.min_height)
    max_height = yaml_config.get('max_height', config.max_height)
    min_arity = yaml_config.get('min_arity', config.min_arity)
    max_arity = yaml_config.get('max_arity', config.max_arity)

    primitives = yaml_config.get('primitives', None)
    func_dict, full_dict, term_dict = parse_primitives(primitives)

    # TODO parse kwargs yaml optionally


def parse_primitives(primitives):
    func_dict = {}
    full_dict = {}
    term_dict = {}

    for prim_key, values in primitives:
        pass
        # TODO parse in type
        # TODO parse rest
        # TODO parse function with fixed kwargs
        # TODO optionally evo kwargs may be here
        # TODO add to propre groups

    return func_dict, full_dict, term_dict
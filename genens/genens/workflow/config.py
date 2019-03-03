# -*- coding: utf-8 -*-

import genens.workflow.model_creation as mc
from genens.gp.types import GpFunctionTemplate, GpTerminalTemplate, TypeArity

from functools import partial


func_config = {
    'cPipe': mc.create_pipeline,
    'dUnion': mc.create_data_union,
    'dTerm': mc.create_empty_pipe
}


full_config = {
    'out': [GpFunctionTemplate('cPipe', [TypeArity('ens', 1), TypeArity('data', (0,1))], 'out')],
    'data': [
        GpFunctionTemplate('dUnion', [TypeArity('data', (2,'n'))], 'data'),
        GpTerminalTemplate('dTerm', 'data')
    ]
}

grow_config = {}
term_config = {}


def estimator_func(est_cls, **kwargs):
    return partial(mc.create_estimator, est_cls=est_cls, const_kwargs=kwargs)


def ensemble_func(ens_cls, **kwargs):
    return partial(mc.create_ensemble, ens_cls=ens_cls, const_kwargs=kwargs)


def ensemble_primitive(ens_name, in_arity, in_type='out', out_type='ens'):
    return GpFunctionTemplate(ens_name, [TypeArity(in_type, in_arity)], out_type)


def predictor_primitive(p_name):
    return GpTerminalTemplate(p_name, 'ens')


def predictor_terminal(p_name):
    return GpTerminalTemplate(p_name, 'out')


def transformer_primitive(t_name, arg_dict=None):
    return GpFunctionTemplate(t_name, [TypeArity('data', 1)], 'data')

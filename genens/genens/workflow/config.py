# -*- coding: utf-8 -*-

from genens.workflow.model_creation import create_pipeline, create_ensemble,\
    create_estimator, create_data_union
from genens.gp.types import FunctionTemplate, GpTerminal, TypeArity

from functools import partial


func_config = {
    'cPipe': create_pipeline,
    'cUnion': create_data_union
}


def estimator_func(est_cls, **kwargs):
    return partial(create_estimator, est_cls=est_cls, const_kwargs=kwargs)


def ensemble_func(ens_cls, **kwargs):
    return partial(create_ensemble, ens_cls=ens_cls, const_kwargs=kwargs)


def ensemble_primitive(ens_name, arg_dict, in_arity, in_type='out'):
    return FunctionTemplate(ens_name, TypeArity(in_type, in_arity), 'ens', kwargs_possible=arg_dict)


def predictor_primitive(p_name, arg_dict):
    return GpTerminal(p_name, 'ens', obj_kwargs=arg_dict)


def transformer_primitive(t_name, arg_dict):
    return FunctionTemplate
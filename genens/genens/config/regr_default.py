# -*- coding: utf-8 -*-

import genens.config.utils as cf

from sklearn import decomposition
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn import ensemble

def create_regr_config():
    config = cf.get_default_config()

    ensembles_func = {
        'ada': cf.ensemble_func(ensemble.AdaBoostRegressor),
        'bagging': cf.ensemble_func(ensemble.BaggingRegressor)
    }

    # TODO
    regr_func = {

    }

    transform_func = {
                "NMF": cf.estimator_func(decomposition.NMF),
                "FA": cf.estimator_func(decomposition.FactorAnalysis),
                "FastICA": cf.estimator_func(decomposition.FastICA),
                "PCA": cf.estimator_func(decomposition.PCA),
                #"KernelPCA" : cf.estimator_func(decomposition.KernelPCA),
                "kBest": cf.estimator_func(feature_selection.SelectKBest),
                "MaxAbsScaler": cf.estimator_func(preprocessing.MaxAbsScaler),
                "MinMaxScaler":cf.estimator_func(preprocessing.MinMaxScaler),
                "Normalizer": cf.estimator_func(preprocessing.Normalizer),
                "StandardScaler": cf.estimator_func(preprocessing.StandardScaler)
    }

    ensemble_kwargs = {
        'ada': {
            'n_estimators': [5, 10, 50, 100, 200],
            'loss': ['linear', 'square', 'exponential']
        },
        'bagging': {
            'n_estimators': [5, 10, 50, 100, 200]
        }
    }

    regr_kwargs = {

    }

    transform_kwargs = {
        'NMF': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'solver': ['cd', 'mu']
        },
        'FA': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        },
        'FastICA': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],

        },
        'PCA': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'whiten': [False, True],
        },
        # 'KernelPCA': {
        #   'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        #   'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        # },
        'kBest': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'score_func': [feature_selection.chi2, feature_selection.f_classif]
        },
        'MaxAbsScaler': {},
        'MinMaxScaler': {},
        "Normalizer": {},
        "StandardScaler": {}
    }

    func_dict = {**ensembles_func, **regr_func, **transform_func}
    kwargs_dict = {**ensemble_kwargs, **regr_kwargs, **transform_kwargs}

    config.add_functions_args(func_dict, kwargs_dict)

    # ensemble config
    config.add_primitive(cf.ensemble_primitive('ada', 1))
    config.add_primitive(cf.ensemble_primitive('bagging', 1))

    # transformer config
    config.add_primitive(cf.transformer_primitive("NMF", 'featsel'))
    config.add_primitive(cf.transformer_primitive("FA", 'featsel'))
    config.add_primitive(cf.transformer_primitive("FastICA", 'featsel'))
    config.add_primitive(cf.transformer_primitive("PCA", 'featsel'))
    #config.add_primitive(cf.transformer_primitive("KernelPCA", 'featsel'))
    config.add_primitive(cf.transformer_primitive("kBest", 'featsel'))

    config.add_primitive(cf.transformer_primitive("MaxAbsScaler", 'scale'))
    config.add_primitive(cf.transformer_primitive("MinMaxScaler", 'scale'))
    config.add_primitive(cf.transformer_primitive("Normalizer", 'scale'))
    config.add_primitive(cf.transformer_primitive("StandardScaler", 'scale'))

    return config

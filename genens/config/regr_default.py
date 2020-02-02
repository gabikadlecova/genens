# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars

from ..config.utils import get_default_config, predictor_primitive, predictor_terminal
from ..config.utils import estimator_func
from ..config.utils import ensemble_func
from ..config.utils import ensemble_primitive
from ..config.utils import transformer_primitive

from sklearn import decomposition
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn import ensemble

import warnings

from ..gp.types import GpFunctionTemplate, TypeArity


def regr_config():
    """
    Creates Genens configuration for regression problems.
    It does not contain regressors, will be extended in future releases.

    :return GenensConfig: Genens configuration for regression problems.
    """

    warnings.warn("This configuration is incomplete. You can extend it with regressors to use it; in next releases,"
                  " it will be finished.")

    config = get_default_config()

    ensembles_func = {
        'voting': ensemble_func(ensemble.VotingRegressor),
        'stacking': ensemble_func(ensemble.StackingRegressor)
    }

    # TODO
    regr_func = {
        'ada': estimator_func(ensemble.AdaBoostRegressor),
        'bagging': estimator_func(ensemble.BaggingRegressor),
        'LinearRegression': estimator_func(LinearRegression),
        'Ridge': estimator_func(Ridge),
        'Lasso': estimator_func(Lasso),
        'ElasticNet': estimator_func(ElasticNet),
        'Lars': estimator_func(Lars)

    }

    transform_func = {
                "NMF": estimator_func(decomposition.NMF),
                "FA": estimator_func(decomposition.FactorAnalysis),
                "FastICA": estimator_func(decomposition.FastICA),
                "PCA": estimator_func(decomposition.PCA),
                #"KernelPCA" : estimator_func(decomposition.KernelPCA),
                "kBest": estimator_func(feature_selection.SelectKBest),
                "MaxAbsScaler": estimator_func(preprocessing.MaxAbsScaler),
                "MinMaxScaler":estimator_func(preprocessing.MinMaxScaler),
                "Normalizer": estimator_func(preprocessing.Normalizer),
                "StandardScaler": estimator_func(preprocessing.StandardScaler)
    }

    ensemble_kwargs = {
        'voting': {
            # 'soft' not included - a lot of classifiers does not support predict_proba
            'voting': ['hard', 'soft']
        },
        'stacking': {
            'stack_method': ['auto', 'predict_proba', 'predict'],
            'cv': [3, 5]
        }
    }

    regr_kwargs = {
        'ada': {
            'n_estimators': [5, 10, 50, 100, 200],
            'loss': ['linear', 'square', 'exponential']
        },
        'bagging': {
            'n_estimators': [5, 10, 50, 100, 200]
        }
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
    config.add_primitive(ensemble_primitive('voting', (2, 'n')))
    config.add_primitive(
        GpFunctionTemplate('stacking',
                           [TypeArity('out', (1, 'n')), TypeArity('out', 1)],
                           'ens',
                           group='ensemble')
    )

    # regressor config
    config.add_primitive(predictor_primitive('ada', 1))
    config.add_primitive(predictor_primitive('bagging', 1))

    config.add_primitive(predictor_primitive('ada', 1), term_only=True)
    config.add_primitive(predictor_primitive('bagging', 1), term_only=True)

    # transformer config
    config.add_primitive(transformer_primitive("NMF", 'featsel'))
    config.add_primitive(transformer_primitive("FA", 'featsel'))
    config.add_primitive(transformer_primitive("FastICA", 'featsel'))
    config.add_primitive(transformer_primitive("PCA", 'featsel'))
    #config.add_primitive(transformer_primitive("KernelPCA", 'featsel'))
    config.add_primitive(transformer_primitive("kBest", 'featsel'))

    config.add_primitive(transformer_primitive("MaxAbsScaler", 'scale'))
    config.add_primitive(transformer_primitive("MinMaxScaler", 'scale'))
    config.add_primitive(transformer_primitive("Normalizer", 'scale'))
    config.add_primitive(transformer_primitive("StandardScaler", 'scale'))

    return config

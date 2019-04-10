# -*- coding: utf-8 -*-

import genens.config.utils as cf

from sklearn import decomposition
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn import svm
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors

from sklearn import ensemble


def create_clf_config(group_weights=None):
    # FUNCTIONS

    ensembles_func = {
        'ada': cf.ensemble_func(ensemble.AdaBoostClassifier),
        'bagging': cf.ensemble_func(ensemble.BaggingClassifier),
        'voting': cf.ensemble_func(ensemble.VotingClassifier)
    }

    clf_func = {
        "KNeighbors": cf.estimator_func(neighbors.KNeighborsClassifier),
        "LinearSVC": cf.estimator_func(svm.LinearSVC),
        "SVC": cf.estimator_func(svm.SVC),
        "logR": cf.estimator_func(linear_model.LogisticRegression),
        "Perceptron": cf.estimator_func(linear_model.Perceptron),
        "SGD": cf.estimator_func(linear_model.SGDClassifier),
        "PAC": cf.estimator_func(linear_model.PassiveAggressiveClassifier),
        "LDA": cf.estimator_func(discriminant_analysis.LinearDiscriminantAnalysis),
        "QDA": cf.estimator_func(discriminant_analysis.QuadraticDiscriminantAnalysis),
        "MLP": cf.estimator_func(neural_network.MLPClassifier),
        "gaussianNB": cf.estimator_func(naive_bayes.GaussianNB),
        "DT": cf.estimator_func(tree.DecisionTreeClassifier),
        "gradBoosting": cf.estimator_func(ensemble.GradientBoostingClassifier),
        "randomForest": cf.estimator_func(ensemble.RandomForestClassifier)
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
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'bagging': {
            'n_estimators': [5, 10, 50, 100, 200]
        },
        'voting': {
            # ['hard', 'soft'] a lot of classifiers does not support predict_proba
            'voting': ['hard']
        }
    }

    clf_kwargs = {
        'KNeighbors': {
            'n_neighbors': [1, 2, 5],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'LinearSVC': {
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l1', 'l2'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'tol': [0.0001, 0.001, 0.01]
        },
        'SVC': {
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'gamma': ['scale', 0.0001, 0.001, 0.01, 0.1, 0.5],
            'tol': [0.0001, 0.001, 0.01]
        },
        'logR': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'tol': [0.0001, 0.001, 0.01],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        'Perceptron': {
            'penalty': ['None', 'l2', 'l1', 'elasticnet'],
            'n_iter': [1, 2, 5, 10, 100],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'SGD': {
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'max_iter': [10, 100, 200],
            'tol': [0.0001, 0.001, 0.01],
            'alpha': [0.0001, 0.001, 0.01],
            'l1_ratio': [0, 0.15, 0.5, 1],
            'epsilon': [0.01, 0.05, 0.1, 0.5],
            'learning_rate': ['constant', 'optimal'],
            'eta0': [0.01, 0.1, 0.5],  # wild guess
            'power_t': [0.1, 0.5, 1, 2]  # dtto
        },
        'PAC': {
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15]
        },
        'LDA': {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5, 1.0]
        },
        'QDA': {
            'reg_param': [0.0, 0.1, 0.5, 1],
            'tol': [0.0001, 0.001, 0.01]
        },
        'MLP': {
            'activation': ['identity', 'logistic', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'tol': [0.0001, 0.001, 0.01],
            'max_iter': [10, 100, 200],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'power_t': [0.1, 0.5, 1, 2],
            'momentum': [0.1, 0.5, 0.9],
            'hidden_layer_sizes': [(100,), (50,), (20,), (10,)]
        },
        'DT': {
            'criterion': ['gini', 'entropy'],
            'max_features': [0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'max_depth': [1, 2, 5, 10, 15, 25, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10, 20]
        },
        'gaussianNB': {},
        'gradBoosting': {
            'loss': ['deviance', 'exponential'],
            'n_estimators': [20, 50, 100, 200, 300],
            'subsample': [0.3, 0.5, 0.75, 1.0]
            # TODO
        },
        'randomForest': {
            'n_estimators': [10, 50, 100, 150, 200]
            # TODO
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
        #'KernelPCA': {
        #   'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        #   'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        #},
        'kBest': {
            'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
            'score_func': [feature_selection.chi2, feature_selection.f_classif]
        },
        'MaxAbsScaler': {},
        'MinMaxScaler': {},
        "Normalizer": {},
        "StandardScaler": {}
    }

    func_dict = {**ensembles_func, **clf_func, **transform_func}
    kwargs_dict = {**ensemble_kwargs, **clf_kwargs, **transform_kwargs}

    if group_weights is None:
        group_weights = {
            'pipeline': 1.0,
            'union': 0.3,
            'prepro': 1.0,
            'ensemble': 0.5,
            'predictor': 1.0,
            'ensemble_l': 1.2,  # lightweight ensembles
            'transform': 1.0
        }

    # add to config
    config = cf.get_default_config()
    config.add_functions_args(func_dict, kwargs_dict)

    # PRIMITIVES

    # ensemble config
    config.add_primitive(cf.ensemble_primitive('ada', 1))
    config.add_primitive(cf.ensemble_primitive('bagging', 1))
    config.add_primitive(cf.ensemble_primitive('voting', (2,'n'), group='ensemble_l'))

    # classifier config
    config.add_primitive(cf.predictor_primitive("KNeighbors"))
    config.add_primitive(cf.predictor_primitive("LinearSVC"))
    config.add_primitive(cf.predictor_primitive("SVC"))
    config.add_primitive(cf.predictor_primitive("logR"))
    config.add_primitive(cf.predictor_primitive("Perceptron"))
    config.add_primitive(cf.predictor_primitive("SGD"))
    config.add_primitive(cf.predictor_primitive("PAC"))
    config.add_primitive(cf.predictor_primitive("gaussianNB"))
    config.add_primitive(cf.predictor_primitive("DT"))
    config.add_primitive(cf.predictor_primitive("gradBoosting"))
    config.add_primitive(cf.predictor_primitive("randomForest"))

    # terminals used only as leaves
    config.add_primitive(cf.predictor_terminal("KNeighbors"), term_only=True)
    config.add_primitive(cf.predictor_terminal("LinearSVC"), term_only=True)
    config.add_primitive(cf.predictor_terminal("SVC"), term_only=True)
    config.add_primitive(cf.predictor_terminal("logR"), term_only=True)
    config.add_primitive(cf.predictor_terminal("Perceptron"), term_only=True)
    config.add_primitive(cf.predictor_terminal("SGD"), term_only=True)
    config.add_primitive(cf.predictor_terminal("PAC"), term_only=True)
    config.add_primitive(cf.predictor_terminal("gaussianNB"), term_only=True)
    config.add_primitive(cf.predictor_terminal("DT"), term_only=True)
    config.add_primitive(cf.predictor_terminal("gradBoosting"), term_only=True)
    config.add_primitive(cf.predictor_terminal("randomForest"), term_only=True)

    # transformer config

    # can be logically ordered via default cData node
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

    # terminals used only as leaves
    config.add_primitive(cf.transformer_primitive("NMF", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("FA", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("FastICA", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("PCA", 'data'), term_only=True)
    # config.add_primitive(cf.transformer_primitive("KernelPCA", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("kBest", 'data'), term_only=True)

    config.add_primitive(cf.transformer_primitive("MaxAbsScaler", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("MinMaxScaler", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("Normalizer", 'data'), term_only=True)
    config.add_primitive(cf.transformer_primitive("StandardScaler", 'data'), term_only=True)

    config.group_weights = group_weights  # checks for missing values according to config terminals

    return config

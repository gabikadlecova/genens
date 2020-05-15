# -*- coding: utf-8 -*-

from genens.config.utils import ensemble_func
from genens.config.utils import estimator_func
from genens.config.utils import get_default_config
from genens.config.utils import ensemble_primitive
from genens.config.utils import predictor_primitive
from genens.config.utils import predictor_terminal
from genens.config.utils import transformer_primitive

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


def clf_config(group_weights=None):
    """
    Creates Genens configuration for classifier problems.

    :param dict group_weights: Dictionary of group weights.
           If not provided, default setting is used.

    :return GenensConfig: Genens configuration for classifier problems.
    """

    # FUNCTIONS
    ensembles_func = {
        'voting': ensemble_func(ensemble.VotingClassifier)
    }

    clf_func = {
        'ada': estimator_func(ensemble.AdaBoostClassifier),
        'bagging': estimator_func(ensemble.BaggingClassifier),
        "KNeighbors": estimator_func(neighbors.KNeighborsClassifier),
        "LinearSVC": estimator_func(svm.LinearSVC),
        "SVC": estimator_func(svm.SVC),
        "logR": estimator_func(linear_model.LogisticRegression),
        "Perceptron": estimator_func(linear_model.Perceptron),
        "SGD": estimator_func(linear_model.SGDClassifier),
        "PAC": estimator_func(linear_model.PassiveAggressiveClassifier),
        "LDA": estimator_func(discriminant_analysis.LinearDiscriminantAnalysis),
        "QDA": estimator_func(discriminant_analysis.QuadraticDiscriminantAnalysis),
        "MLP": estimator_func(neural_network.MLPClassifier),
        "gaussianNB": estimator_func(naive_bayes.GaussianNB),
        "DT": estimator_func(tree.DecisionTreeClassifier),
        "gradBoosting": estimator_func(ensemble.GradientBoostingClassifier),
        "randomForest": estimator_func(ensemble.RandomForestClassifier),
        "extraTrees": estimator_func(ensemble.ExtraTreesClassifier)
    }

    transform_func = {
            "NMF": estimator_func(decomposition.NMF),
            "FA": estimator_func(decomposition.FactorAnalysis),
            "FastICA": estimator_func(decomposition.FastICA),
            "PCA": estimator_func(decomposition.PCA),
            # "KernelPCA" : estimator_func(decomposition.KernelPCA),
            "kBest": estimator_func(feature_selection.SelectKBest),
            "MaxAbsScaler": estimator_func(preprocessing.MaxAbsScaler),
            "MinMaxScaler": estimator_func(preprocessing.MinMaxScaler),
            "Normalizer": estimator_func(preprocessing.Normalizer),
            "StandardScaler": estimator_func(preprocessing.StandardScaler)
    }

    ensemble_kwargs = {
        'voting': {
            'voting': ['soft']
            #'voting': ['hard', 'soft']
        }
    }

    clf_kwargs = {
        'ada': {
            'n_estimators': [5, 10, 50, 100, 200],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'bagging': {
            'n_estimators': [5, 10, 50, 100, 200]
        },
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
            'tol': [0.0001, 0.001, 0.01],
            'probability': [True, False]
        },
        'logR': {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
            'tol': [0.0001, 0.001, 0.01],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        'Perceptron': {
            'penalty': ['None', 'l2', 'l1', 'elasticnet'],
            'max_iter': [1, 2, 5, 10, 100],
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
            'n_estimators': [20, 50, 100, 200],
            'subsample': [0.3, 0.5, 0.75, 1.0]
            # TODO
        },
        'randomForest': {
            'n_estimators': [10, 50, 100, 150, 200]
            # TODO
        },
        'extraTrees': {
            'n_estimators': [10, 50, 100, 150, 200]
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
        #    'feat_frac': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        #    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
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

    func_dict = {**ensembles_func, **clf_func, **transform_func}
    kwargs_dict = {**ensemble_kwargs, **clf_kwargs, **transform_kwargs}

    if group_weights is None:
        group_weights = {
            "pipeline": 1.0,
            # "union": 0.3,
            "prepro": 1.0,
            "ensemble": 0.5,
            "predictor": 1.0,
            "transform": 1.0
        }

    # add to config
    config = get_default_config()
    config.add_functions_args(func_dict, kwargs_dict)

    # PRIMITIVES

    # ensemble config
    config.add_primitive(ensemble_primitive('voting', (2, 'n')))

    # classifier config
    config.add_primitive(predictor_primitive('ada'))
    config.add_primitive(predictor_primitive('bagging'))
    config.add_primitive(predictor_primitive("KNeighbors"))
    config.add_primitive(predictor_primitive("LinearSVC"))
    config.add_primitive(predictor_primitive("SVC"))
    config.add_primitive(predictor_primitive("logR"))
    config.add_primitive(predictor_primitive("Perceptron"))
    config.add_primitive(predictor_primitive("SGD"))
    config.add_primitive(predictor_primitive("PAC"))
    config.add_primitive(predictor_primitive("gaussianNB"))
    config.add_primitive(predictor_primitive("DT"))
    config.add_primitive(predictor_primitive("gradBoosting"))
    config.add_primitive(predictor_primitive("randomForest"))
    config.add_primitive(predictor_primitive("extraTrees"))

    # terminals used only as leaves
    config.add_primitive(predictor_terminal("KNeighbors"), term_only=True)
    config.add_primitive(predictor_terminal("LinearSVC"), term_only=True)
    config.add_primitive(predictor_terminal("SVC"), term_only=True)
    config.add_primitive(predictor_terminal("logR"), term_only=True)
    config.add_primitive(predictor_terminal("Perceptron"), term_only=True)
    config.add_primitive(predictor_terminal("SGD"), term_only=True)
    config.add_primitive(predictor_terminal("PAC"), term_only=True)
    config.add_primitive(predictor_terminal("gaussianNB"), term_only=True)
    config.add_primitive(predictor_terminal("DT"), term_only=True)
    config.add_primitive(predictor_terminal("gradBoosting"), term_only=True)
    config.add_primitive(predictor_terminal("randomForest"), term_only=True)
    config.add_primitive(predictor_terminal("extraTrees"), term_only=True)

    # transformer config

    # can be logically ordered via default cData node
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

    # terminals used only as leaves
    config.add_primitive(transformer_primitive("NMF", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("FA", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("FastICA", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("PCA", 'data'), term_only=True)
    # config.add_primitive(transformer_primitive("KernelPCA", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("kBest", 'data'), term_only=True)

    config.add_primitive(transformer_primitive("MaxAbsScaler", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("MinMaxScaler", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("Normalizer", 'data'), term_only=True)
    config.add_primitive(transformer_primitive("StandardScaler", 'data'), term_only=True)

    config.group_weights = group_weights  # checks for missing values according to config terminals

    return config

# -*- coding: utf-8 -*-
import os

from sklearn import feature_selection
from genens.config.genens_config import GenensConfig, parse_config

file_dir = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_PATH = file_dir + '/config_default.yaml'
BASE_CLF_PATH = file_dir + '/config_clf.yaml'


def default_raw_config() -> GenensConfig:
    return parse_config(BASE_CONFIG_PATH, evo_kwargs=default_kwargs())


def clf_default_config() -> GenensConfig:
    default_config = default_raw_config()
    return parse_config(BASE_CLF_PATH, base_config=default_config, evo_kwargs=clf_kwargs())


def default_kwargs() -> dict:
    return {k: {} for k in ['cPred', 'cPipe', 'cData', 'cFeatSelect', 'cScale']}


def clf_kwargs() -> dict:
    """
    Creates Genens hyperparameter configuration for classifier problems.

    Returns:
        dict: Hyperparameter configuration for classifier problems.
    """

    ensemble_kwargs = {
        "voting": {
            "voting": ["hard", "soft"]
        },
        "stacking": {
            "cv": [None, 3, 7],
            "passthrough": [True, False]
        }
    }

    clf_kwargs = {
        "ada": {
            "n_estimators": [5, 10, 50, 100, 200],
            "algorithm": ["SAMME", "SAMME.R"],
            "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0]
        },
        "bagging": {
            "n_estimators": [5, 10, 50, 100, 200],
            "oob_score": [True, False],
            "max_samples": [0.2, 0.5, 0.8, 1.0],
            "max_features": [0.2, 0.5, 0.8, 1.0]
        },
        "KNeighbors": {
            "n_neighbors": [1, 2, 5],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        },
        "LinearSVC": {
            "loss": ["hinge", "squared_hinge"],
            "penalty": ["l1", "l2"],
            "C": [0.1, 0.5, 1.0, 2, 5, 10, 15],
            "tol": [0.0001, 0.001, 0.01]
        },
        "SVC": {
            "C": [0.1, 0.5, 1.0, 2, 5, 10, 15],
            "gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 0.5],
            "tol": [0.0001, 0.001, 0.01],
            "probability": [True, False]
        },
        "logR": {
            "penalty": ["l1", "l2"],
            "C": [0.1, 0.5, 1.0, 2, 5, 10, 15],
            "tol": [0.0001, 0.001, 0.01],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        },
        "Perceptron": {
            "penalty": ["None", "l2", "l1", "elasticnet"],
            "max_iter": [1, 2, 5, 10, 100],
            "alpha": [0.0001, 0.001, 0.01]
        },
        "SGD": {
            "penalty": ["none", "l2", "l1", "elasticnet"],
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            "max_iter": [10, 100, 200],
            "tol": [0.0001, 0.001, 0.01],
            "alpha": [0.0001, 0.001, 0.01],
            "l1_ratio": [0, 0.15, 0.5, 1],
            "epsilon": [0.01, 0.05, 0.1, 0.5],
            "learning_rate": ["constant", "optimal"],
            "eta0": [0.01, 0.1, 0.5],  # wild guess
            "power_t": [0.1, 0.5, 1, 2]  # dtto
        },
        "PAC": {
            "loss": ["hinge", "squared_hinge"],
            "C": [0.1, 0.5, 1.0, 2, 5, 10, 15]
        },
        "LDA": {
            "solver": ["lsqr", "eigen"],
            "shrinkage": [None, "auto", 0.1, 0.5, 1.0]
        },
        "QDA": {
            "reg_param": [0.0, 0.1, 0.5, 1],
            "tol": [0.0001, 0.001, 0.01]
        },
        "MLP": {
            "activation": ["identity", "logistic", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "tol": [0.0001, 0.001, 0.01],
            "max_iter": [10, 100, 200],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "power_t": [0.1, 0.5, 1, 2],
            "momentum": [0.1, 0.5, 0.9],
            "hidden_layer_sizes": [(100,), (50,), (20,), (10,)]
        },
        "DT": {
            "criterion": ["gini", "entropy"],
            "max_features": [0.05, 0.1, 0.25, 0.5, 0.75, 1],
            "max_depth": [1, 2, 5, 10, 15, 25, 50, 100],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10, 20]
        },
        "gaussianNB": {},
        "gradBoosting": {
            "loss": ["deviance", "exponential"],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            "n_estimators": [20, 50, 100],
            "subsample": [0.3, 0.5, 0.75, 1.0],
            "max_depth": [None, 3, 5, 10, 20, 50, 100],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10, 20],
        },
        "randomForest": {
            "n_estimators": [10, 50, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_features": [None, "sqrt", 0.05, 0.2, 0.5, 0.8],
            "max_depth": [None, 3, 5, 10, 20, 50, 100],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.001, 0.01, 0.03, 0.05],
            "oob_score": [True, False]
        },
        "extraTrees": {
            "n_estimators": [10, 50, 100, 150],
            "criterion": ["gini", "entropy"],
            "max_features": [None, "sqrt", 0.05, 0.2, 0.5, 0.8],
            "max_depth": [None, 3, 5, 10, 20, 50, 100],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.001, 0.01, 0.03, 0.05],
            "oob_score": [True, False]
        }
    }

    transform_kwargs = {
        "NMF": {
            "feat_frac": [0.1, 0.5, 0.8, None],
            "solver": ["cd", "mu"],
            "tol": [0.0001, 0.001, 0.01]
        },
        "FA": {
            "feat_frac": [0.1, 0.5, 0.8, None]
        },
        "FastICA": {
            "feat_frac": [0.1, 0.5, 0.8, None],
            "algorithm": ["parallel", "deflation"]
        },
        "PCA": {
            "feat_frac": [None, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
            "n_components": ["mle", 2, 3],
            "whiten": [False, True],
        },
        # "KernelPCA": {
        #    "feat_frac": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        #    "kernel": ["linear", "poly", "rbf", "sigmoid", "cosine"]
        # },
        "kBest": {
            "feat_frac": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
            "score_func": [feature_selection.chi2, feature_selection.f_classif]
        },
        "MaxAbsScaler": {},
        "MinMaxScaler": {},
        "Normalizer": {},
        "StandardScaler": {}
    }

    kwargs_dict = {**ensemble_kwargs, **clf_kwargs, **transform_kwargs}
    return kwargs_dict

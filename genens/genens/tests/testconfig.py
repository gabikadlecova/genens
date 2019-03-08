# -*- coding: utf-8 -*-

import random
import genens.config.utils as cf
import genens.workflow.model_creation as mc
import genens.render.graph as graph

from genens.gp.operators import gen_half

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def create_test_config():
    func_config = cf.func_config

    full_config = cf.full_config
    grow_config = cf.grow_config
    term_config = cf.term_config

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
        "DT": cf.estimator_func(tree.DecisionTreeClassifier)
    }

    prep_func = {
            "NMF": cf.estimator_func(decomposition.NMF),
            "FA": cf.estimator_func(decomposition.FactorAnalysis),
            "FastICA": cf.estimator_func(decomposition.FastICA),
            "PCA": cf.estimator_func(decomposition.PCA),
            "kBest": cf.estimator_func(feature_selection.SelectKBest),
            "MaxAbsScaler": cf.estimator_func(preprocessing.MaxAbsScaler),
            "MinMaxScaler":cf.estimator_func(preprocessing.MinMaxScaler),
            "Normalizer": cf.estimator_func(preprocessing.Normalizer),
            "StandardScaler": cf.estimator_func(preprocessing.StandardScaler)
    }

    func_config = {**func_config, **ensembles_func, **clf_func, **prep_func}

    full_config['ens'].append(cf.ensemble_primitive('ada', 1))
    full_config['ens'].append(cf.ensemble_primitive('bagging', 1))
    full_config['ens'].append(cf.ensemble_primitive('voting', (2,'n')))

    grow_config['ens'].append(cf.predictor_primitive("KNeighbors"))
    grow_config['ens'].append(cf.predictor_primitive("LinearSVC"))
    grow_config['ens'].append(cf.predictor_primitive("SVC"))
    grow_config['ens'].append(cf.predictor_primitive("logR"))
    grow_config['ens'].append(cf.predictor_primitive("Perceptron"))
    grow_config['ens'].append(cf.predictor_primitive("SGD"))
    grow_config['ens'].append(cf.predictor_primitive("PAC"))
    grow_config['ens'].append(cf.predictor_primitive("gaussianNB"))
    grow_config['ens'].append(cf.predictor_primitive("DT"))

    term_config['out'].append(cf.predictor_terminal("KNeighbors"))
    term_config['out'].append(cf.predictor_terminal("LinearSVC"))
    term_config['out'].append(cf.predictor_terminal("SVC"))
    term_config['out'].append(cf.predictor_terminal("logR"))
    term_config['out'].append(cf.predictor_terminal("Perceptron"))
    term_config['out'].append(cf.predictor_terminal("SGD"))
    term_config['out'].append(cf.predictor_terminal("PAC"))
    term_config['out'].append(cf.predictor_terminal("gaussianNB"))
    term_config['out'].append(cf.predictor_terminal("DT"))

    full_config['data'].append(cf.transformer_primitive("NMF"))
    full_config['data'].append(cf.transformer_primitive("FA"))
    full_config['data'].append(cf.transformer_primitive("FastICA"))
    full_config['data'].append(cf.transformer_primitive("PCA"))
    full_config['data'].append(cf.transformer_primitive("kBest"))
    full_config['data'].append(cf.transformer_primitive("MaxAbsScaler"))
    full_config['data'].append(cf.transformer_primitive("MinMaxScaler"))
    full_config['data'].append(cf.transformer_primitive("Normalizer"))
    full_config['data'].append(cf.transformer_primitive("StandardScaler"))

    kwargs_dict = {
        'cPipe': {

        },
        'dUnion': {

        },
        'dTerm': {

        },
        'NMF': {
            'n_components': [2,3,4,5,7,9],
            'solver': ['cd', 'mu']
        },
        'FA': {
            'n_components': [2,3,4,5,7,9],
        },
        'FastICA': {
            'n_components': [2,3,4,5,7,9],

        },
        'KernelPCA': {
            'n_components': [2,3,4,5,7,9],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        },

        'ada': {
            'n_estimators': [5, 10, 50, 100, 200],
            'algorithm': ['SAMME']
        },
        'bagging': {
            'n_estimators': [5, 10, 50, 100, 200]
        },
        'voting' : {

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
        'MaxAbsScaler': {

        },
        'MinMaxScaler': {

        },
        "Normalizer": {

        },
        "StandardScaler": {

        },

        'PCA': {
            'n_components': [2,3,4,5,7,9],
            'whiten': [False, True],
        },
        'kBest': {
            'k': [1,2,5],
            'score_func': [feature_selection.chi2, feature_selection.f_classif]
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
            'n_iter': [5, 10, 100],
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
        'gaussianNB': {}
    }

    data, target = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.33, random_state=42)

    random.seed(42)

    it = gen_half(20, full_config, grow_config, term_config, kwargs_dict, 5, 4)
    for i, res_tree in enumerate(it):
        graph.create_graph(res_tree, "tree{}.png".format(i))
        wf = mc.create_workflow(res_tree, func_config)

        try:
            wf.fit(X_train, y_train)
            pred = wf.predict(X_test)
        except Exception as e:
            print(e)
            continue

        print("Score {}: {}".format(i, accuracy_score(y_test, pred)))


if __name__ == "__main__":
    create_test_config()
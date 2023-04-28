import argparse
import numpy as np
import random

from sklearn.metrics import accuracy_score

from genens.config.config_templates import clf_default_config

from genens.config.genens_config import GenensConfig, parse_config

from genens import GenensClassifier
from genens.workflow.evaluate import SampleCrossValEvaluator, CrossValEvaluator
from genens.tests.load_datasets import load_dataset


def run_test(kwarg_dict):
    random_state = kwarg_dict.pop('random_state')
    np.random.seed(random_state)
    random.seed(random_state)

    dataset = kwarg_dict.pop('dataset')
    X, y = load_dataset(dataset)

    cv_k = kwarg_dict['cv_k']
    size = kwarg_dict['size']

    if size == 1.0:
        evaluator = CrossValEvaluator(cv_k=cv_k)
    else:
        evaluator = SampleCrossValEvaluator(cv_k=cv_k, sample_size=size, per_gen=True)

    config_kwargs = {
        p: kwarg_dict[p] for p in ['max_height', 'max_arity', 'cx_pb', 'mut_pb', 'mut_args_pb', 'mut_node_pb']
    }
    config = parse_config(config_kwargs, base_config=clf_default_config())

    clf = GenensClassifier(
        config=config,
        evaluator=evaluator,
        n_jobs=kwarg_dict['n_jobs'],
        pop_size=kwarg_dict['pop_size'],
        n_gen=kwarg_dict['n_gen'],
        timeout=kwarg_dict['timeout']
    )

    clf.fit(X, y, verbose=2)
    return clf, X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--random_state', default=42, type=int)

    parser.add_argument('--n_jobs', default=-1, type=int)
    parser.add_argument('--pop_size', default=200, type=int)
    parser.add_argument('--n_gen', default=15, type=int)

    parser.add_argument('--max_height', default=4, type=int)
    parser.add_argument('--max_arity', default=3, type=int)

    parser.add_argument('--cx_pb', default=0.5, type=float)
    parser.add_argument('--mut_pb', default=0.3, type=float)
    parser.add_argument('--mut_args_pb', default=0.6, type=float)
    parser.add_argument('--mut_node_pb', default=0.3, type=float)

    parser.add_argument('--cv_k', default=3, type=int)
    parser.add_argument('--size', default=0.25, type=float)
    parser.add_argument('--timeout', default=240, type=int)
    parser.add_argument('--log_path', default=None)

    args = parser.parse_args()

    clf, X, y = run_test(vars(args))

    print(accuracy_score(y, clf.predict(X)))

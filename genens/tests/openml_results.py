# -*- coding: utf-8 -*-

import openml
import pandas as pd


def get_score_stats():

    benchmark_suite = openml.study.get_study('OpenML-CC18', 'tasks')

    accuracy_df = pd.DataFrame(columns=['task_id', 'name', 'accuracy'])

    for task_id in benchmark_suite.tasks:
        evaluations = openml.evaluations.list_evaluations('predictive_accuracy', task=[task_id])

        vals = [(e.task_id, e.data_name, e.value) for e in evaluations.values()]
        df = pd.DataFrame(vals, columns=['task_id', 'name', 'accuracy'])

        accuracy_df = accuracy_df.append(df)

        print('Task {}'.format(task_id))

    return accuracy_df


if __name__ == "__main__":
    get_score_stats()
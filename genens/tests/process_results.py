# -*- coding: utf-8 -*-

import openml
import os
import pandas as pd


def get_score_stats(columns, dirs):

    df = pd.DataFrame()
    for col_name, dir_path in zip(columns, dirs):
        file_paths = os.listdir(dir_path)
        file_paths = ["/".join(dir_path, f) for f in file_paths]

        df[col_name] = read_score_list(file_paths)

    return df


def read_score_list(*args):
    score_list = []
    for file_name in args:
        max_val = 0.0
        with open(file_name, 'r') as in_file:
            for line in in_file:
                max_val = max(line, max_val)

        score_list.append(max_val)
    return score_list


def get_openml_stats():

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
    df = get_openml_stats()
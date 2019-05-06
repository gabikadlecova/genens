# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import openml
import os
import pandas as pd
import seaborn as sns


def get_score_stats(columns, dirs):

    df = pd.DataFrame()
    for col_name, dir_path in zip(columns, dirs):
        file_paths = os.listdir(dir_path)
        file_paths = ["/".join([dir_path, f]) for f in file_paths]

        df[col_name] = read_score_list(file_paths)

    return df


def read_score_list(file_list):
    score_list = []
    for file_name in file_list:
        max_val = 0.0
        with open(file_name, 'r') as in_file:
            for line in in_file:
                val = float(line)
                max_val = max(val, max_val)

        score_list.append(max_val)
    return score_list


def boxplot_compare_columns(df, out_path, out_name='outbox.png'):
    bx_plot = sns.boxplot(x='variable', y='value', data=pd.melt(df))

    fig = bx_plot.get_figure()
    fig.savefig('/'.join([out_path, out_name]))


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
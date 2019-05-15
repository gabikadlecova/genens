# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
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


def boxplot_get_stats(df):
    res = []
    for column in df.columns:
        bx = plt.boxplot(df[column], notch=True)

        c_min = np.min(df[column])
        c_max = np.max(df[column])

        bx_data = bx['boxes'][0].get_ydata()
        c_cilo = bx_data[2]
        c_med = bx_data[3]
        c_cihi = bx_data[4]

        res.append([c_min, c_cilo, c_med, c_cihi, c_max])

    res_df = pd.DataFrame(data=res, columns=['minimum', 'ci_low', 'median', 'ci_high', 'maximum'], index=df.columns)
    return res_df.round(4)


def boxplot_compare_columns(df, out_path, out_name='outbox.png', x_axis='Evaluation strategy',
                            eval_metric='Predictive accuracy'):
    sns.set()

    bx_plot = sns.boxplot(x='variable', y='value', data=pd.melt(df), notch=True)
    plt.xlabel(x_axis)
    plt.ylabel(eval_metric)

    plt.tight_layout()
    plt.savefig('/'.join([out_path, out_name]))


def get_openml_stats():

    benchmark_suite = openml.study.get_study('OpenML-CC18', 'tasks')

    accuracy_df = pd.DataFrame(columns=['task_id', 'name', 'accuracy'])

    for task_id in benchmark_suite.tasks:
        print('Task {}'.format(task_id))

        evaluations = openml.evaluations.list_evaluations('predictive_accuracy', task=[task_id])

        vals = [(e.task_id, e.data_name, e.value) for e in evaluations.values()]
        df = pd.DataFrame(vals, columns=['task_id', 'name', 'accuracy'])

        accuracy_df = accuracy_df.append(df)

    return accuracy_df


if __name__ == "__main__":
    df = get_openml_stats()
    df.to_csv("openml-data.csv", sep=';')

# -*- coding: utf-8 -*-
"""
Module for visualization of the evolution process.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


def export_plot(estimator, out_file):
    sns.set()

    gen = estimator.logbook.select("gen")
    score_max = estimator.logbook.chapters["score"].select("max")
    score_avg = estimator.logbook.chapters["score"].select("avg")

    fig = plt.figure()
    fig.suptitle("Evolution of population scores")

    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score")

    line1 = ax1.plot(gen, score_max, label='Maximum Score')
    line2 = ax1.plot(gen, score_avg, label='Average Score')

    lines = line1 + line2

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.savefig(out_file)
    plt.close()

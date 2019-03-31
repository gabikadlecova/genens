# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt


def export_plot(estimator, out_file):
    sns.set()

    gen = estimator.logbook.select("gen")
    score_max = estimator.logbook.chapters["score"].select("max")
    test_max = estimator.logbook.chapters["test_score"].select("max")
    score_avg = estimator.logbook.chapters["score"].select("avg")
    test_avg = estimator.logbook.chapters["test_score"].select("avg")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score")

    line1 = ax1.plot(gen, score_max, label='Maximum Score')
    line2 = ax1.plot(gen, score_avg, label='Average Score')

    line3 = ax1.plot(gen, test_max, label='Maximum Test score')
    line4 = ax1.plot(gen, test_avg, label='Average Test score')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.savefig(out_file)
    plt.close()

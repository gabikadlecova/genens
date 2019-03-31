# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def export_plot(estimator, out_file):
    gen = estimator.logbook.select("gen")
    max_vals = estimator.logbook.chapters["score"].select("max")
    test_vals = estimator.logbook.chapters["test_score"].select("max")

    plt.plot(gen, max_vals)
    plt.plot(gen, test_vals)
    plt.savefig(out_file)
    plt.close()

# -*- coding:utf-8 -*-
"""Metric summary output."""

import numpy as np


def _mean_and_sample_std(values):
    """Use sample SD for independent runs; a singleton has SD 0 by definition."""
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.size == 0:
        raise ValueError("Cannot summarize an empty metric list.")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, Ensemble=False):
    accuracy_mean, accuracy_std = _mean_and_sample_std(Accuracy_List)
    precision_mean, precision_std = _mean_and_sample_std(Precision_List)
    recall_mean, recall_std = _mean_and_sample_std(Recall_List)
    auc_mean, auc_std = _mean_and_sample_std(AUC_List)
    prc_mean, prc_std = _mean_and_sample_std(AUPR_List)

    if not Ensemble:
        print("The model's results:")
        filepath = "./{}/results.txt".format(DATASET)
    else:
        print("The ensemble model's results:")
        filepath = "./{}/ensemble_results.txt".format(DATASET)

    lines = [
        'Accuracy(std):{:.4f}({:.4f})'.format(accuracy_mean, accuracy_std),
        'Precision(std):{:.4f}({:.4f})'.format(precision_mean, precision_std),
        'Recall(std):{:.4f}({:.4f})'.format(recall_mean, recall_std),
        'AUC(std):{:.4f}({:.4f})'.format(auc_mean, auc_std),
        'PRC(std):{:.4f}({:.4f})'.format(prc_mean, prc_std),
    ]
    with open(filepath, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

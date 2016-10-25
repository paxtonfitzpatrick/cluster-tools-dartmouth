#!/usr/bin/python

import numpy as np
import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sb
from config import config

def parse_fname(fname):
    fname = fname[len('xvalresults_'):]
    x = [i for i, char in enumerate(fname) if char == '_']
    assert(len(x) == 1)
    x = x[0]

    try:
        w = int(fname[0:x])
        mu = float(fname[x+1:-4])
    except:
        mu = []
        w = []
    return np.array(mu), np.array(w)


fig_dir = os.path.join(config['resultsdir'], 'figs')
save_file = os.path.join(config['resultsdir'], 'paramsearch_results.pkl')

if not os.path.isfile(save_file):
    results = list()
    mus = np.array([])
    windowlengths = np.array([])

    for file in os.listdir(config['resultsdir']):
        if fnmatch.fnmatch(file, 'xvalresults_*_*.npz'):
            mu, w = parse_fname(file)
            if mu.size == 0:
                continue

            results.append(file)
            mus = np.append(mus, mu)
            windowlengths = np.append(windowlengths, w)

    results = pd.DataFrame({'errors': np.array(map((lambda x: np.load(os.path.join(config['resultsdir'], x))['results'].tolist()['error']), results)),
                       'accuracies': np.array(map((lambda x: np.load(os.path.join(config['resultsdir'], x))['results'].tolist()['accuracy']), results)),
                       'ranks': np.array(map((lambda x: np.load(os.path.join(config['resultsdir'], x))['results'].tolist()['rank']), results)),
                       'mu': mus,
                       'windowlength': windowlengths})
    results.to_pickle(save_file)

results = pd.read_pickle(save_file)

# compile results
xval_ranks = results.pivot('windowlength', 'mu', 'ranks')
xval_errors = results.pivot('windowlength', 'mu', 'errors')
xval_accuracies = results.pivot('windowlength', 'mu', 'accuracies')

# make fig_dir if it doesn't already exist
try:
    os.stat(fig_dir)
except:
    os.makedirs(fig_dir)

# print out cross validation figures
plt.close()
xval_ranks_fig = sb.heatmap(xval_ranks)
xval_ranks_fig.get_figure().savefig(os.path.join(fig_dir, 'xval_ranks_fig.pdf'))

plt.close()
xval_errors_fig = sb.heatmap(xval_errors)
xval_errors_fig.get_figure().savefig(os.path.join(fig_dir, 'xval_errors_fig.pdf'))

plt.close()
xval_accuracies_fig = sb.heatmap(xval_accuracies)
xval_accuracies_fig.get_figure().savefig(os.path.join(fig_dir, 'xval_accuracies_fig.pdf'))


# best parameters are the ones that yield the highest classification accuracy
accuracies = xval_accuracies.values
best_results = results[results['accuracies'] == np.max(results['accuracies'])]
best_mu = round(best_results['mu'], 10)
best_windowlength = int(best_results['windowlength'])

np.savez(os.path.join(config['resultsdir'], 'best_parameters'), windowlength=best_windowlength, mu=best_mu)



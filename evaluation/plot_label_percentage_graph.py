import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 16
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (5,4)
GRAPH_FORMAT = 'pdf'
GRAPH_TITLE = 'Metrics as a Function of Sequence Length'
TARGET_EPOCH = 10
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
NUM_METRICS = len(METRICS_NAMES)
METRICS_SYMBOLS = ['o', '^', 'v']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue', 'coral']

# load experiments data
exp_dir = "/mnt/walkure_public/deanz/models/bracelet/feb25_label_percentage_exp_seq2fc"
exp_files = os.listdir(exp_dir)
exp_metric_files = {}
for exp_file in exp_files:
    csv_path = os.path.join(exp_dir, exp_file, 'metrics_mean_subject.csv')
    df = pd.read_csv(csv_path)
    exp_metric_files[exp_file] = df.values[:, 2:]

# collect all metric values for each sequence length
lp_metric = {}
for exp_file in exp_files:
    lp_value = float(exp_file.split('lp')[-1])
    if lp_value not in lp_metric:
        lp_metric[lp_value] = []
    lp_metric[lp_value].append(exp_metric_files[exp_file])

# extract mean and std values for each sequence length
lp_mean = {}
lp_std = {}
for lp_val in lp_metric:
    if len(lp_metric[lp_val]) != 10:
        print(f"WARNING! Sequence length {lp_val} has {len(lp_metric[lp_val])} experiments.")
    lp_mean[lp_val] = np.mean(lp_metric[lp_val], axis=0)
    lp_std[lp_val] = np.std(lp_metric[lp_val], axis=0)

# iterate ovev sequence lengths and extract into an ordered array
lp_vals = sorted(list(lp_metric.keys()))
metrics_means = np.zeros((len(lp_vals), NUM_METRICS))
metrics_stds = np.zeros((len(lp_vals), NUM_METRICS))
for i, lp_val in enumerate(lp_vals):
    metrics_means[i, :] = lp_mean[lp_val]
    metrics_stds[i, :] = lp_std[lp_val]

# plot graph
eps = 0.005
offsets = [-1*eps, 0, 1*eps]
for i in range(NUM_METRICS):
    plt.scatter([x+offsets[i] for x in lp_vals], metrics_means[:, i], marker=METRICS_SYMBOLS[i], s=80, color=COLORS[i])
    plt.plot([x+offsets[i] for x in lp_vals], metrics_means[:, i], color=COLORS[i])

    # plot std
    lower_bound = np.clip(metrics_means[:, i] - metrics_stds[:, i], 0, 1)
    upper_bound = np.clip(metrics_means[:, i] + metrics_stds[:, i], 0, 1)
    for j in range(len(lp_vals)):
        plt.plot([lp_vals[j]+offsets[i]]*2, [lower_bound[j], upper_bound[j]], color=COLORS[i], alpha=0.5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, marker=y, label=z, linewidth=3, markersize=8, linestyle='-') for x,y,z in zip(COLORS, METRICS_SYMBOLS,METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower left', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(lp_vals, fontsize=FONT_SIZE-6)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([lp_vals[0] - 0.025, lp_vals[-1] + 0.025])
plt.xlabel(r'$\sigma$', fontsize=FONT_SIZE)
plt.ylabel('Score')

# save plot
plt.grid(True)
plt.tight_layout(pad=0.1)
plt.savefig("label_percentage_graph_metrics.{}".format(GRAPH_FORMAT))
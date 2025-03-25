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
TARGET_EPOCH = 50

METRICS_NAMES = ['Accuracy','Recall', 'Precision']
NUM_METRICS = len(METRICS_NAMES)
METRICS_SYMBOLS = ['o', '^', 'v']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue', 'coral']

# load experiments data
exp_dir = "/mnt/walkure_public/deanz/models/bracelet/feb26_generalization_exp_2fc"
exp_files = os.listdir(exp_dir)
exp_metric_files = {}
for exp_file in exp_files:
    nst_value = int(exp_file.split('nst')[-1])
    csv_path = os.path.join(exp_dir, exp_file, 'metrics_mean_subject.csv')
    df = pd.read_csv(csv_path)
    exp_metric_files[exp_file] = df.values[:, 2:]

# collect all metric values for each sequence length 
nst_metric = {}
for exp_file in exp_files:
    nst_value = int(exp_file.split('nst')[-1])
    if nst_value not in nst_metric:
        nst_metric[nst_value] = []
    nst_metric[nst_value].append(exp_metric_files[exp_file])

# extract mean and std values for each sequence length
nst_mean = {}
nst_std = {}
for nst_value in nst_metric:
    if len(nst_metric[nst_value]) != 10:
        print(f"WARNING! NST value {nst_value} has {len(nst_metric[nst_value])} experiments.")
    nst_mean[nst_value] = np.mean(nst_metric[nst_value], axis=0)
    nst_std[nst_value] = np.std(nst_metric[nst_value], axis=0)

# iterate ovev sequence lengths and extract into an ordered array
nst_values = sorted(list(nst_metric.keys()))
metrics_means = np.zeros((len(nst_values), NUM_METRICS))
metrics_stds = np.zeros((len(nst_values), NUM_METRICS))
for i, nst_value in enumerate(nst_values):
    metrics_means[i, :] = nst_mean[nst_value]
    metrics_stds[i, :] = nst_std[nst_value]

# plot graph
eps = 0.025
offsets = [-1*eps, 0, 1*eps]
for i in range(NUM_METRICS):
    plt.scatter([x+offsets[i] for x in nst_values], metrics_means[:, i], marker=METRICS_SYMBOLS[i], s=80, color=COLORS[i])
    plt.plot([x+offsets[i] for x in nst_values], metrics_means[:, i], color=COLORS[i])

    # plot std
    lower_bound = np.clip(metrics_means[:, i] - metrics_stds[:, i], 0, 1)
    upper_bound = np.clip(metrics_means[:, i] + metrics_stds[:, i], 0, 1)
    for j in range(len(nst_values)):
        plt.plot([nst_values[j]+offsets[i]]*2, [lower_bound[j], upper_bound[j]], color=COLORS[i], alpha=0.5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, marker=y, label=z, linewidth=3, markersize=8, linestyle='-') for x,y,z in zip(COLORS, METRICS_SYMBOLS,METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower right', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(nst_values, fontsize=FONT_SIZE-6)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([nst_values[0] - 0.2, nst_values[-1] + 0.2])
plt.xlabel('Number of Subjects in Train Set')
plt.ylabel('Score')

# save plot
plt.grid(True)
plt.tight_layout(pad=0.1)
plt.savefig("nst_graph_metrics.{}".format(GRAPH_FORMAT))
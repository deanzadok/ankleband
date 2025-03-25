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
SENSOR_FREQ = 200
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
NUM_METRICS = len(METRICS_NAMES)
METRICS_SYMBOLS = ['o', '^', 'v']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue', 'coral']

# load experiments data
exp_dir = "/mnt/walkure_public/deanz/models/bracelet/feb24_sequence_exp_seq2fc"
exp_files = os.listdir(exp_dir)
exp_metric_files = {}
for exp_file in exp_files:
    csv_path = os.path.join(exp_dir, exp_file, 'metrics_mean_subject.csv')
    df = pd.read_csv(csv_path)
    exp_metric_files[exp_file] = df.values[:, 2:]

# collect all metric values for each sequence length 
seq_length_metric = {}
for exp_file in exp_files:
    append_value = int(exp_file.split('append')[-1])
    if append_value not in seq_length_metric:
        seq_length_metric[append_value] = []
    seq_length_metric[append_value].append(exp_metric_files[exp_file])

# extract mean and std values for each sequence length
seq_length_mean = {}
seq_length_std = {}
for seq_length in seq_length_metric:
    if len(seq_length_metric[seq_length]) != 10:
        print(f"WARNING! Sequence length {seq_length} has {len(seq_length_metric[seq_length])} experiments.")
    seq_length_mean[seq_length] = np.mean(seq_length_metric[seq_length], axis=0)
    seq_length_std[seq_length] = np.std(seq_length_metric[seq_length], axis=0)

# iterate ovev sequence lengths and extract into an ordered array
seq_lengths = sorted(list(seq_length_metric.keys()))
metrics_means = np.zeros((len(seq_lengths), NUM_METRICS))
metrics_stds = np.zeros((len(seq_lengths), NUM_METRICS))
for i, seq_length in enumerate(seq_lengths):
    metrics_means[i, :] = seq_length_mean[seq_length]
    metrics_stds[i, :] = seq_length_std[seq_length]

# convert sequence length to time duration
durations = [(x*2 / SENSOR_FREQ) for x in seq_lengths]

# plot graph
eps = 0.01
offsets = [-1*eps, 0, 1*eps]
for i in range(NUM_METRICS):
    plt.scatter([x+offsets[i] for x in durations], metrics_means[:, i], marker=METRICS_SYMBOLS[i], s=80, color=COLORS[i])
    plt.plot([x+offsets[i] for x in durations], metrics_means[:, i], color=COLORS[i])

    # plot std
    lower_bound = np.clip(metrics_means[:, i] - metrics_stds[:, i], 0, 1)
    upper_bound = np.clip(metrics_means[:, i] + metrics_stds[:, i], 0, 1)
    for j in range(len(durations)):
        plt.plot([durations[j]+offsets[i]]*2, [lower_bound[j], upper_bound[j]], color=COLORS[i], alpha=0.5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, marker=y, label=z, linewidth=3, markersize=8, linestyle='-') for x,y,z in zip(COLORS, METRICS_SYMBOLS,METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower left', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(durations, fontsize=FONT_SIZE-6)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([durations[0] - 0.025, durations[-1] + 0.025])
plt.ylim([0.19, 0.99])
plt.xlabel('Duration of Input Vector [s]')
plt.ylabel('Score')

# save plot
plt.grid(True)
plt.tight_layout(pad=0.1)
plt.savefig("seq_length_graph_metrics.{}".format(GRAPH_FORMAT))
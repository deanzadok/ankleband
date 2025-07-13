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
MAX_FREQ = 200
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
NUM_METRICS = len(METRICS_NAMES)
METRICS_SYMBOLS = ['o', '^', 'v']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue', 'coral']
GRAPH_FORMAT = 'pdf'

# load experiments data
exp_dir = "/mnt/walkure_public/deanz/models/bracelet/jul10_frequency_exp_c1d_bn_im64_2fc_regsampling_bs64"
exp_files = os.listdir(exp_dir)
exp_metric_files = {}
for exp_file in exp_files:
    csv_path = os.path.join(exp_dir, exp_file, 'metrics.csv')
    df = pd.read_csv(csv_path)
    exp_metric_files[exp_file] = df.iloc[-1].values[1:]

# collect all metric values for each sequence length 
freq_metric = {}
for exp_file in exp_files:
    step_value = int(exp_file.split('_step')[-1])
    freq_value = MAX_FREQ // step_value
    if freq_value not in freq_metric:
        freq_metric[freq_value] = []
    freq_metric[freq_value].append(exp_metric_files[exp_file])

# extract mean and std values for each sequence length
freq_mean = {}
freq_std = {}
for freq_value in freq_metric:
    if len(freq_metric[freq_value]) != 10:
        print(f"WARNING! NST value {freq_value} has {len(freq_metric[freq_value])} experiments.")
    freq_mean[freq_value] = np.mean(freq_metric[freq_value], axis=0)
    freq_std[freq_value] = np.std(freq_metric[freq_value], axis=0)

# iterate ovev sequence lengths and extract into an ordered array
freq_values = sorted(list(freq_metric.keys()))
metrics_means = np.zeros((len(freq_values), NUM_METRICS))
metrics_stds = np.zeros((len(freq_values), NUM_METRICS))
for i, freq_value in enumerate(freq_values):
    metrics_means[i, :] = freq_mean[freq_value]
    metrics_stds[i, :] = freq_std[freq_value]

# plot graph
eps = 0.0125
offsets = [-1*eps, 0, 1*eps]
for i in range(NUM_METRICS):
    # plot mean values as a function of the log value of the frequency
    plt.scatter(np.log(freq_values)+offsets[i], metrics_means[:, i], marker=METRICS_SYMBOLS[i], s=80, color=COLORS[i])
    plt.plot(np.log(freq_values)+offsets[i], metrics_means[:, i], color=COLORS[i])    

    # plot std
    lower_bound = np.clip(metrics_means[:, i] - metrics_stds[:, i], 0, 1)
    upper_bound = np.clip(metrics_means[:, i] + metrics_stds[:, i], 0, 1)
    for j in range(len(freq_values)):
        plt.plot([np.log(freq_values[j])+offsets[i]]*2, [lower_bound[j], upper_bound[j]], color=COLORS[i], alpha=0.5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, marker=y, label=z, linewidth=3, markersize=8, linestyle='-') for x,y,z in zip(COLORS, METRICS_SYMBOLS,METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower right', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
xticks_vals = np.log(freq_values)
xticks_vals[0] = xticks_vals[0] - 0.025
plt.xticks(xticks_vals, freq_values, fontsize=FONT_SIZE-6)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([np.log(freq_values[0]) - 0.05, np.log(freq_values[-1]) + 0.05])
plt.xlabel('Input Data Frequency (Log Scale) [Hz]')
plt.ylabel('Score')

# save plot
plt.grid(True)
plt.tight_layout(pad=0.1)
plt.savefig("freq_graph_metrics.{}".format(GRAPH_FORMAT))
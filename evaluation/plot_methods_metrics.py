import sys
sys.path.append('.')
sys.path.append('..')
import os
import pandas as pd
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FONT_SIZE = 16
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams['axes.axisbelow'] = True
METHODS_NAMES = ['LDA', 'SVM', 'DTW+MLP', 'RF', 'Ours']
METHODS_COLORS = ['indianred', 'gold', 'mediumturquoise', 'deepskyblue', 'mediumpurple']
METHODS_STD_COLORS = ['firebrick', 'goldenrod', 'lightseagreen', 'steelblue', 'rebeccapurple']
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
GRAPH_FORMAT = 'pdf'

# experiment folders
exp_dirs = {
    'classic_lda'                                       : '/mnt/walkure_public/deanz/models/bracelet/jun26_classic_lda_exp', 
    'classic_lsvm'                                       : '/mnt/walkure_public/deanz/models/bracelet/jul10_classic_lsvc_exp',
    'dtw_mlp_20cands_norm'                              : '/mnt/walkure_public/deanz/models/bracelet/mar16_dtw_20cands_mlp_exp_normalized',
    'classic_rf'                                        : '/mnt/walkure_public/deanz/models/bracelet/jul10_classic_rf_n8d8_exp',
    'jul10_c1d_bn_im64_2fc_regsampling_bs64'            : '/mnt/walkure_public/deanz/models/bracelet/jul10_c1d_bn_im64_2fc_regsampling_bs64'
    }

EXP_NAMES = list(exp_dirs.keys())

# iterate over experiment folders and load metrics
metrics_scores = {}
for exp_name, exp_dir in exp_dirs.items():
    # iterate over all subfolders under the experiment directory
    accuracy_scores, recall_scores, precision_scores = [], [], []
    for exp_subject_dir in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, exp_subject_dir)):
            # construct the path to the metrics CSV file
            cm_csv_path = os.path.join(exp_dir, exp_subject_dir, 'metrics.csv')
            if os.path.isfile(cm_csv_path):
                # read the CSV file and append to metrics_df
                df = pd.read_csv(cm_csv_path)
                
                # take the 20th row if len(df) > 1
                if len(df) > 1:
                    df = df.iloc[9]
                    accuracy_scores.append(df['Accuracy'])
                    recall_scores.append(df['Recall'])
                    precision_scores.append(df['Precision'])
                else:
                    accuracy_scores.append(df['Accuracy'].values[0])
                    recall_scores.append(df['Recall'].values[0])
                    precision_scores.append(df['Precision'].values[0])

    # store mean and std for each metric
    metrics_scores[exp_name] = [[np.nanmean(accuracy_scores), np.nanstd(accuracy_scores)],
                                [np.nanmean(recall_scores), np.nanstd(recall_scores)],
                                [np.nanmean(precision_scores), np.nanstd(precision_scores)]]    

# plot graph
bar_width = 0.15
std_line_hat_size = 0.09
y_start = 0.0
step_size = 0.9
eps = 0.1
x_start = step_size - eps - 0.2
x = x_start
x_text = step_size
metric_step = bar_width * 1.0

for k, metric_name in enumerate(METRICS_NAMES):
    for j, method_name in enumerate(METHODS_NAMES):
        
        metric_features = metrics_scores[EXP_NAMES[j]][k]

        # draw bar for a single metric
        bar_val = plt.bar(x, metric_features[0], color=METHODS_COLORS[j], width=bar_width)

        # draw std line
        if metric_features[0] >= 0 and metric_features[0] <= 1:                
            lower_bound = max(0, metric_features[0] - metric_features[1])
            upper_bound = min(1, metric_features[0] + metric_features[1])
            plt.plot([x]*2, [lower_bound, upper_bound], color=METHODS_STD_COLORS[j])

            # draw the upper and lower hats of the std line
            plt.plot([x - std_line_hat_size/2, x + std_line_hat_size/2], [upper_bound]*2, color=METHODS_STD_COLORS[j])
            plt.plot([x - std_line_hat_size/2, x + std_line_hat_size/2], [lower_bound]*2, color=METHODS_STD_COLORS[j])

        # write values as text
        if metric_features[0] >= 0 and metric_features[0] <= 1:
            text_val = f"{metric_features[0]:.2f}"
            text_y_loc = metric_features[0] + metric_features[1] + 0.05
            text_x_loc = x - 0.09
        else:
            text_val = 'N/A'
            text_y_loc = 0.055
            text_x_loc = x - 0.05
        plt.text(text_x_loc, text_y_loc, text_val, color=METHODS_STD_COLORS[j], fontsize=FONT_SIZE-6)

        x += bar_width
    x += metric_step
    x_text += step_size

# prepare customized legend
legend_handles = [plt.scatter([], [], color=color, marker='s', s=82) for color in METHODS_COLORS]
metrics_legend = plt.legend(legend_handles, METHODS_NAMES, loc='lower left', framealpha=0.5, fontsize=FONT_SIZE-4)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.tick_params(left = False, bottom = False)

plt.xticks([x_start + (len(METHODS_NAMES)/2 * bar_width - (0.5 * bar_width)) + x * (metric_step + (len(METHODS_NAMES) * bar_width)) for x in range(len(METRICS_NAMES))], METRICS_NAMES)
plt.tick_params(axis='x', which='both', labelsize=FONT_SIZE)

plt.yticks(np.arange(0,1.25, step=0.25))
plt.tick_params(axis='y', which='both', labelsize=FONT_SIZE-4)

plt.xlim([x_start - 0.15, x_start + len(METRICS_NAMES) * (len(METHODS_NAMES) * bar_width + metric_step) - metric_step])
plt.ylim([0.20, 1.075])

# save plot
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("metrics_cigraph.{}".format(GRAPH_FORMAT)))
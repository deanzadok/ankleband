import sys
sys.path.append('.')
sys.path.append('..')
import os
import pandas as pd
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FONT_SIZE = 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (13.5,2.75)
plt.rcParams['axes.axisbelow'] = True
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue']
TEXT_COLORS = ['seagreen', 'goldenrod', 'royalblue']
GRAPH_FORMAT = 'pdf'

# load confusion matrices from experiments folders data
exp_dir = '/mnt/walkure_public/deanz/models/bracelet/jul10_lp_exp_c1d_bn_im64_2fc_regsampling_bs64'
exp_files = [f for f in os.listdir(exp_dir) if 'lp0.5' in f] # iterate only over folders with lp 0.5 in their name
metrics_df = []
for exp_file in exp_files:
    csv_path = os.path.join(exp_dir, exp_file, 'metrics.csv')
    subject_idx = int(exp_file.split('_s')[-1].split('_')[0])

    df = pd.read_csv(csv_path)
    df['subject'] = subject_idx
    metrics_df.append(df.iloc[-1:])

# concat and prepare subjects
subjects_df = pd.concat(metrics_df).drop(columns=['Unnamed: 0'])
subjects_list = sorted(subjects_df['subject'].unique().tolist())

eps = 0.285
bar_width = eps

# draw bars for recall and precision
plt.bar([x - eps for x in subjects_list], [subjects_df[subjects_df['subject'] == x][METRICS_NAMES[0]].values[0] for x in subjects_list], color=COLORS[0], width=bar_width, alpha=0.85)
plt.bar([x for x in subjects_list], [subjects_df[subjects_df['subject'] == x][METRICS_NAMES[1]].values[0] for x in subjects_list], color=COLORS[1], width=bar_width, alpha=0.85)
plt.bar([x + eps for x in subjects_list], [subjects_df[subjects_df['subject'] == x][METRICS_NAMES[2]].values[0] for x in subjects_list], color=COLORS[2], width=bar_width, alpha=0.85)
for subject_idx in subjects_list:
    accuracy_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[0]].values[0]:.2f}"
    recall_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[1]].values[0]:.2f}"
    precision_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[2]].values[0]:.2f}"
    plt.text(subject_idx - eps - 0.15, subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[0]].values[0] + 0.01, accuracy_text_val, color=TEXT_COLORS[0], fontsize=FONT_SIZE-5)
    plt.text(subject_idx - 0.155, subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[1]].values[0] + 0.01, recall_text_val, color=TEXT_COLORS[1], fontsize=FONT_SIZE-5)
    plt.text(subject_idx - 0.15 + eps, subjects_df[subjects_df['subject'] == subject_idx][METRICS_NAMES[2]].values[0] + 0.01, precision_text_val, color=TEXT_COLORS[2], fontsize=FONT_SIZE-5)

# prepare customized legend
lines = [Line2D([0], [0], color=x, label=z, linewidth=7, linestyle='-') for x,z in zip(COLORS, METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower left', framealpha=0.5, fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(subjects_list, fontsize=FONT_SIZE-4)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([subjects_list[0] - 2 * eps, subjects_list[-1] + 2 * eps])
plt.ylim([0.65, 1.01])
plt.xlabel('Subject', fontsize=FONT_SIZE)
plt.ylabel('Score', fontsize=FONT_SIZE)

# save plot
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join("subjects_metrics_bars.{}".format(GRAPH_FORMAT)))
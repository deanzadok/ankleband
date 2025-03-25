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
SUBJECTS_NAMES = ['01', '04', '05', '06', '10', '17', '19', '20', '21', '23', '24', '25']
METRICS_SYMBOLS = ['o', '^', 'v']
METRICS_NAMES = ['Accuracy','Recall', 'Precision']
COLORS = ['mediumseagreen', 'gold', 'cornflowerblue']
TEXT_COLORS = ['seagreen', 'goldenrod', 'royalblue']
NUM_EXPS = 5
GRAPH_FORMAT = 'pdf'

# load confusion matrices from experiments folders data
exp_dir = '/mnt/walkure_public/deanz/models/bracelet/feb25_label_percentage_exp_seq2fc'
exp_files = [f for f in os.listdir(exp_dir) if 'lp0.5' in f] # iterate only over folders with lp 0.5 in their name
metrics_df = []
for exp_file in exp_files:
    cm_csv_path = os.path.join(exp_dir, exp_file, 'metrics_mean_subject.csv')
    metrics_df.append(pd.read_csv(cm_csv_path))

# concat and prepare subjects
subjects_df = pd.concat(metrics_df).drop(columns=['Unnamed: 0'])
subjects_list = sorted(subjects_df['subject'].unique().tolist())

eps = 0.25
bar_width = 0.25

# draw bars for recall and precision
plt.bar([x - eps for x in subjects_list], [subjects_df[subjects_df['subject'] == x]['accuracy'].values[0] for x in subjects_list], color=COLORS[0], width=bar_width)
plt.bar([x for x in subjects_list], [subjects_df[subjects_df['subject'] == x]['recall'].values[0] for x in subjects_list], color=COLORS[1], width=bar_width)
plt.bar([x + eps for x in subjects_list], [subjects_df[subjects_df['subject'] == x]['precision'].values[0] for x in subjects_list], color=COLORS[2], width=bar_width)
for subject_idx in subjects_list:
    accuracy_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx]['accuracy'].values[0]:.2f}"
    recall_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx]['recall'].values[0]:.2f}"
    precision_text_val = f"{subjects_df[subjects_df['subject'] == subject_idx]['precision'].values[0]:.2f}"
    plt.text(subject_idx - 0.115 - eps, subjects_df[subjects_df['subject'] == subject_idx]['accuracy'].values[0] + 0.01, accuracy_text_val, color=TEXT_COLORS[0], fontsize=FONT_SIZE-6)
    plt.text(subject_idx - 0.1, subjects_df[subjects_df['subject'] == subject_idx]['recall'].values[0] + 0.01, recall_text_val, color=TEXT_COLORS[1], fontsize=FONT_SIZE-6)
    plt.text(subject_idx - 0.1 + eps, subjects_df[subjects_df['subject'] == subject_idx]['precision'].values[0] + 0.01, precision_text_val, color=TEXT_COLORS[2], fontsize=FONT_SIZE-6)

# prepare customized legend
lines = [Line2D([0], [0], color=x, label=z, linewidth=7, linestyle='-') for x,z in zip(COLORS, METRICS_NAMES)]
metrics_legend = plt.legend(handles=lines, loc='lower right', fontsize=FONT_SIZE-2)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(subjects_list, fontsize=FONT_SIZE-4)
plt.yticks(fontsize=FONT_SIZE-4)
plt.xlim([subjects_list[0] - 2 * eps, subjects_list[-1] + 1.75])
plt.ylim([0.70, 0.99])
plt.xlabel('Subject', fontsize=FONT_SIZE)
plt.ylabel('Score', fontsize=FONT_SIZE)

# save plot
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("subjects_metrics_bars.{}".format(GRAPH_FORMAT)))
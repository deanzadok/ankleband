import sys
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import matplotlib.pyplot as plt
import seaborn as sns

GRAPH_FORMAT = 'pdf'
FONT_SIZE = 18
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (5,4)

def format_large_numbers(value):
    """formats large numbers with abbreviations (K, M, etc.)."""
    if value >= 1000000:
        return f"{value / 1000000:.1f}M"
    elif value >= 1000:
        return f"{value / 1000:.1f}K"
    else:
        return str(value)

# load confusion matrices from experiments folders data
exp_dir = '/mnt/walkure_public/deanz/models/bracelet/feb25_label_percentage_exp_seq2fc'
exp_files = [f for f in os.listdir(exp_dir) if 'lp0.5' in f] # iterate only over folders with lp 0.5 in their name
confusion_matrices_df = []
for exp_file in exp_files:
    cm_csv_path = os.path.join(exp_dir, exp_file, 'confusion_matrix_classes.csv')
    confusion_matrices_df.append(pd.read_csv(cm_csv_path))

# sum up values into a single dataframe
cm_df = pd.concat(confusion_matrices_df).groupby(level=0).sum()
cm_df.drop(columns=['Unnamed: 0'], inplace=True)

# apply the formatting function to the values
cm_df_formatted = cm_df.map(format_large_numbers)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=cm_df_formatted, fmt="", cmap="icefire", cbar=False) #fmt="" allows string annotations
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(fontsize=FONT_SIZE-2)
plt.yticks(fontsize=FONT_SIZE-2)
plt.tight_layout() # adjust layout to prevent labels from overlapping

# save confusion matrix
plt.savefig("cm_classes.{}".format(GRAPH_FORMAT), dpi=300) # save with a higher DPI for better image quality
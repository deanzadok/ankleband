import sys
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
from torch import optim, nn
from data.load_data import DataManagement, TorchDatasetManagement
from trainer.utils import ConfigManager, initiate_model, MMDLoss
from sklearn.metrics import recall_score, precision_score, accuracy_score
from trainer.models.conv1d_model import Conv1DNet
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/bracelet/test_regular_bracelet.json', type=str)
parser.add_argument('--visualize_cm', '-visualize_cm', help="if to visualize the confusion matrix or not", default=False, type=bool)
args = parser.parse_args()

# load config file
cfg = ConfigManager(json_name=args.json)

# define gpu
device = torch.device('cuda:{}'.format(cfg.system.gpu) if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print('WARNING! GPU is unavailable.')

# check if output folder exists
if not os.path.isdir(cfg.output_dir):
    os.makedirs(cfg.output_dir)

# load train and test datasets
data_mng = DataManagement(cfg=cfg)
test_dataset = TorchDatasetManagement(cfg=cfg, data_df=data_mng.test_df, inputs_names_stacked=data_mng.inputs_names_stacked, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=True)

# create models
model = Conv1DNet(cfg=cfg)
model.load_state_dict(torch.load(cfg.model.weights, map_location=f'cuda:{cfg.system.gpu}'))
model = model.to(device)
model.eval()

print('Started testing...')
labels_list, preds_list = [], []
for i_test, test_batch in enumerate(test_dataloader):

    # filter batch
    test_inputs, test_labels = test_batch

    # forward pass
    test_inputs = test_inputs.to(device)
    test_predictions = model(test_inputs)

    # store labels and predictions in lists
    labels_list.append(test_labels.numpy())
    preds_list.append(np.argmax(torch.nn.functional.softmax(test_predictions.detach(), dim=1).cpu().numpy(), axis=1))

# concatenate labels and predictions
labels = np.concatenate(labels_list)
preds = np.concatenate(preds_list)

# compute confusion matrix
cm = confusion_matrix(labels, preds)
class_names = [str(x) for x in range(5)] # provide class names
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# save confusion matrix dataframe
cm_df.to_csv(os.path.join(cfg.output_dir, 'confusion_matrix_classes.csv'))

# create confusion matrix plot
if args.visualize_cm:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False) # cbar=False removes colorbar
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if needed
    plt.tight_layout() # adjust layout to prevent labels from overlapping

    # create timestamped filename and save
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # YearMonthDay_HourMinuteSecond
    filename = f"cm_{timestamp}.png"
    plt.savefig(os.path.join(cfg.output_dir, filename), dpi=300) # save with a higher DPI for better image quality
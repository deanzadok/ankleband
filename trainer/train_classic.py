import sys
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from data.load_data import DataManagement, TorchDatasetManagement
from trainer.utils import ConfigManager, initiate_model
from sklearn.metrics import recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('always') # "error", "ignore", "always", "default", "module" or "once"

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/bracelet/regular_bracelet_classic.json', type=str)
parser.add_argument('--loo', '-loo', help='override leave one subject out', default=None, type=int)
args = parser.parse_args()

# load config file
cfg = ConfigManager(json_name=args.json)
if args.loo is not None:
    cfg.data.leave_subject_out = args.loo
    # replace subject id in output dir
    new_output_dir = cfg.output_dir.split('/')[-1].split('_s')[0] + "_s" + f"{args.loo:02}" + '_' \
                     + '_'.join(cfg.output_dir.split('/')[-1].split('_s')[-1].split('_')[-1:])
    cfg.output_dir = os.path.join(os.path.dirname(cfg.output_dir), new_output_dir)

# define gpu
device = torch.device('cuda:{}'.format(cfg.system.gpu) if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print('WARNING! GPU is unavailable.')

# check if output folder exists
if not os.path.isdir(cfg.output_dir):
    os.makedirs(cfg.output_dir)

# load train and test datasets
data_mng = DataManagement(cfg=cfg)

# create model and train
print('Started training...')
model = initiate_model(cfg=cfg)
model.fit(data_mng.train_X, data_mng.train_y)

# compute predictions on test set and compute metrics
test_preds = model.predict(data_mng.test_X)
accuracy_scores_test_mean = accuracy_score(data_mng.test_y, test_preds)
recall_scores_test_mean = recall_score(data_mng.test_y, test_preds, average='macro')
precision_scores_test_mean = precision_score(data_mng.test_y, test_preds, average='macro')

# print results
print('Accuracy: {:.3f}, Recall: {:.3f}, Precision: {:.3f}'.format(accuracy_scores_test_mean, recall_scores_test_mean, precision_scores_test_mean))

# store metrics values in output dir
metrics_df = pd.DataFrame({'Accuracy': [accuracy_scores_test_mean], 'Recall': [recall_scores_test_mean], 'Precision': [precision_scores_test_mean]})
metrics_df.to_csv(os.path.join(cfg.output_dir, 'metrics.csv'))






import sys
sys.path.append('.')
sys.path.append('..')
import os
import argparse
import numpy as np
import pandas as pd
import torch
from data.load_data import DataManagement, TorchDatasetManagement
from trainer.utils import ConfigManager
from sklearn.metrics import recall_score, precision_score, accuracy_score
from trainer.models.conv1d_model import Conv1DNet
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/bracelet/test_regular_bracelet.json', type=str)
parser.add_argument('--loo', '-loo', help='override leave one subject out', default=None, type=int)
parser.add_argument('--append', '-append', help='override number of samples to append for training', default=None, type=int)
parser.add_argument('--label_percentage', '-label_percentage', help='override gesture percentage to be selected for training.', default=None, type=float)
parser.add_argument('--nst', '-nst', help='override force number of subjects for training.', default=None, type=int)
parser.add_argument('--step', '-step', help='override size of step to change frequency of data training', default=None, type=int)
args = parser.parse_args()

# load config file
cfg = ConfigManager(json_name=args.json)
if args.loo is not None:

    # replace leave one out subject in output dir and model weights
    subject_options = [f"_s{x:02}_" for x in range(1,11)]
    for option in subject_options:
        if option in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.replace(option, f"_s{args.loo:02}_")
            cfg.model.weights = cfg.model.weights.replace(option, f"_s{args.loo:02}_")
            break

    cfg.data.leave_subject_out = args.loo

if args.append is not None:

    append_options = [f'_append{x}' for x in [12, 13, 15, 17, 20, 24, 30, 40, 60, 120]][::-1]
    # append_options = [f'_append{x}' for x in range(10, 160, 10)]
    for option in append_options:
        if option in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.replace(option, f'_append{args.append}')
            cfg.model.weights = cfg.model.weights.replace(option, f'_append{args.append}')
            break

    cfg.data.append = args.append

if args.label_percentage is not None:

    lp_options = [f"_lp{x:03}" for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    for option in lp_options:
        if option in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.replace(option, f"_lp{args.label_percentage}")
            cfg.model.weights = cfg.model.weights.replace(option, f"_lp{args.label_percentage}")
            break

    cfg.data.label_percentage = args.label_percentage

if args.nst is not None:

    nst_options = [f"_nst{x}" for x in range(1,10)]
    for option in nst_options:
        if option in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.replace(option, f"_nst{args.nst}")
            cfg.model.weights = cfg.model.weights.replace(option, f"_nst{args.nst}")
            break

    cfg.data.force_num_subjects_train = args.nst

if args.step is not None:

    step_options = [f"_step{x}" for x in range(1,11)]
    for option in step_options:
        if option in cfg.output_dir:
            cfg.output_dir = cfg.output_dir.replace(option, f"_step{args.step}")
            cfg.model.weights = cfg.model.weights.replace(option, f"_step{args.step}")
            break

    cfg.data.step = args.step


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
test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.batch_num_workers)

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

# compute metrics and store them
labels = np.concatenate(labels_list)
preds = np.concatenate(preds_list)
accuracy_scores_test_mean = accuracy_score(labels, preds)
recall_scores_test_mean = recall_score(labels, preds, average='macro')
precision_scores_test_mean = precision_score(labels, preds, average='macro')

# save the metrics in a dataframe
data = {'subject': int(cfg.output_dir.split('/')[-1].split('_s')[1].split('_')[0]),
        'accuracy': [accuracy_scores_test_mean],
        'recall': [recall_scores_test_mean],
        'precision': [precision_scores_test_mean]}
metrics_df = pd.DataFrame(data)

# save confusion matrix dataframe
metrics_df.to_csv(os.path.join(cfg.output_dir, 'metrics_mean_subject.csv'))
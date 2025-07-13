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
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from data.load_data import DataManagement, TorchDatasetManagement
from trainer.utils import ConfigManager, initiate_model
from sklearn.metrics import recall_score, precision_score, accuracy_score
from trainer.models.conv1d_model import Conv1DNet
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('always') # "error", "ignore", "always", "default", "module" or "once"

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/bracelet/regular_bracelet_leaveone.json', type=str)
parser.add_argument('--loo', '-loo', help='override leave one subject out', default=None, type=int)
parser.add_argument('--append', '-append', help='override number of samples to append for training', default=None, type=int)
parser.add_argument('--label_percentage', '-label_percentage', help='override gesture percentage to be selected for training.', default=None, type=float)
parser.add_argument('--nst', '-nst', help='override force number of subjects for training.', default=None, type=int)
parser.add_argument('--step', '-step', help='override size of step to change frequency of data training', default=None, type=int)
args = parser.parse_args()

# load config file
cfg = ConfigManager(json_name=args.json)
if args.loo is not None:
    cfg.data.leave_subject_out = args.loo
    # replace subject id in output dir
    new_output_dir = cfg.output_dir.split('/')[-1].split('_s')[0] + "_s" + f"{args.loo:02}" + '_' \
                     + '_'.join(cfg.output_dir.split('/')[-1].split('_s')[-1].split('_')[-1:])
    cfg.output_dir = os.path.join(os.path.dirname(cfg.output_dir), new_output_dir)
if args.append is not None:
    cfg.data.append = args.append
    cfg.output_dir = cfg.output_dir + f"_append{args.append}"
if args.label_percentage is not None:
    cfg.data.label_percentage = args.label_percentage
    cfg.output_dir = cfg.output_dir + f"_lp{args.label_percentage}"
if args.nst is not None:
    cfg.data.force_num_subjects_train = args.nst
    cfg.output_dir = cfg.output_dir + f"_nst{args.nst}"
if args.step is not None:
    cfg.data.step = args.step
    cfg.output_dir = cfg.output_dir + f"_step{args.step}"

# define gpu
device = torch.device('cuda:{}'.format(cfg.system.gpu) if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print('WARNING! GPU is unavailable.')

# check if output folder exists
if not os.path.isdir(cfg.output_dir):
    os.makedirs(cfg.output_dir)

# load train and test datasets
data_mng = DataManagement(cfg=cfg)
train_dataset = TorchDatasetManagement(cfg=cfg, data_df=data_mng.train_df, inputs_names_stacked=data_mng.inputs_names_stacked, is_train=True)
test_dataset = TorchDatasetManagement(cfg=cfg, data_df=data_mng.test_df, inputs_names_stacked=data_mng.inputs_names_stacked, is_train=False)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.batch_num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.batch_num_workers)

# create model
model = Conv1DNet(cfg=cfg)
if cfg.model.weights != "":
    model.load_state_dict(torch.load(cfg.model.weights))
    model.fc3 = nn.Linear(model.fc3.in_features, cfg.data.classes)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay, eps=cfg.training.epsilon)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, cfg.training.learning_rate, cfg.training.scheduler_steps, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
criterion = nn.CrossEntropyLoss()

print('Started training...')
train_losses_epoch, test_losses_epoch = [], []
test_acc_values, test_rec_values, test_pre_values = [], [], []
for epoch in range(cfg.training.epochs):  # loop over the dataset multiple times
    start_time = time.time()

    # resample datasets each epoch to avoid class overfitting
    if cfg.training.weighted_sampling:
        train_dataset.resample_data()
        test_dataset.resample_data()

    train_losses, test_losses = [], []

    # train on train set
    model.train()
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", unit="batch", leave=False) as pbar:
        for i_train, batch in enumerate(train_dataloader):

            # filter batch
            inputs, labels = batch
            if inputs.size(0) <= 1: # skip if batch size is less than one
                continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.to(device)
            predictions = model(inputs)

            labels = labels.to(device)
            loss = criterion(predictions, labels.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1) # update the tqdm bar
            
    train_losses_epoch.append(np.array(train_losses).mean())

    # evaluate on test set
    model.eval()
    labels_test_set, preds_test_set = [], []
    with tqdm(total=len(test_dataloader), desc=f"Epoch {epoch+1}/{cfg.training.epochs}", unit="batch", leave=False) as pbar:
        for i_test, batch in enumerate(test_dataloader):

            # filter batch
            inputs, labels = batch
            if inputs.size(0) <= 1: # skip if batch size is less than one
                continue

            # forward + backward + optimize
            inputs = inputs.to(device)
            predictions = model(inputs)

            labels = labels.to(device)
            test_loss = criterion(predictions, labels.long())
            test_losses.append(test_loss.item())

            labels_test_set.append(labels.cpu().numpy())
            preds_test_set.append(np.argmax(torch.nn.functional.softmax(predictions.detach(), dim=1).cpu().numpy(), axis=1))

            pbar.set_postfix({"Test Loss": test_loss.item()})
            pbar.update(1) # update the tqdm bar

    test_losses_epoch.append(np.array(test_losses).mean())

    # compute metrics for test set and store them
    labels_test_set = np.concatenate(labels_test_set)
    preds_test_set = np.concatenate(preds_test_set)
    accuracy_scores_test_mean = accuracy_score(labels_test_set, preds_test_set)
    recall_scores_test_mean = recall_score(labels_test_set, preds_test_set, average='macro')
    precision_scores_test_mean = precision_score(labels_test_set, preds_test_set, average='macro')
    test_acc_values.append(accuracy_scores_test_mean)
    test_rec_values.append(recall_scores_test_mean)
    test_pre_values.append(precision_scores_test_mean)

    # printing statistics
    epoch_duration = time.time() - start_time
    print('Epoch: {}, Loss: {:.3f}, Test Loss: {:.3f}, Accuracy: {:.3f}, Recall: {:.3f}, Precision: {:.3f}.'
          .format(epoch+1, np.array(train_losses).mean(), np.array(test_losses).mean(), accuracy_scores_test_mean, recall_scores_test_mean, precision_scores_test_mean))

    # save model
    if (epoch+1) % cfg.training.cp_interval == 0 and epoch > 0:
        print('Saving weights to {}'.format(cfg.output_dir))
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"model_weights_{epoch+1}.pt"))


    # store losses and metrics values in output dir
    losses_df = pd.DataFrame({'Train': train_losses_epoch, 'Test': test_losses_epoch})
    losses_df.to_csv(os.path.join(cfg.output_dir, 'losses.csv'))
    metrics_df = pd.DataFrame({'Accuracy': test_acc_values, 'Recall': test_rec_values, 'Precision': test_pre_values})
    metrics_df.to_csv(os.path.join(cfg.output_dir, 'metrics.csv'))

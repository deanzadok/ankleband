import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    def __init__(self, cfg):
        super(Conv1DNet, self).__init__()
        self.cfg = cfg

        self.in_channels = 6

        self.conv_features = 10

        self.im_size = 64

        self.kernel_a = 3

        self.conv_out_dim = (self.cfg.data.append // self.kernel_a) * self.conv_features

        self.conv1d_1 = nn.Conv1d(self.in_channels, self.conv_features, kernel_size=self.kernel_a, stride=self.kernel_a, padding=0, bias=False)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(self.conv_out_dim) # shape 200

        # create a set of consecutive linear layers using nn.Sequential
        fc_layers = []
        for i in range(self.cfg.model.num_fc_layers - 1):
            if i == 0:
                fc_layers.append(nn.Linear(self.conv_out_dim, self.im_size))
            else:
                fc_layers.append(nn.Linear(self.im_size, self.im_size))
            fc_layers.append(nn.BatchNorm1d(self.im_size))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(self.im_size, self.cfg.data.classes))
        self.fc_layers = nn.Sequential(*fc_layers)

    def extract_features(self, x_input):

        y_conv = self.conv1d_1(x_input)
        conv_features = self.flatten(y_conv)

        return conv_features


    def forward(self, x_input):

        conv_features = self.extract_features(x_input)
        y_features = F.relu(self.bn1(conv_features))

        return self.fc_layers(y_features)

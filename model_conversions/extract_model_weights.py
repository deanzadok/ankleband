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
from trainer.utils import ConfigManager
from trainer.models.tiny_model import TinyNet
from trainer.models.conv1d_model import Conv1DNet
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/bracelet/test_regular_bracelet.json', type=str)
args = parser.parse_args()

# load config file
cfg = ConfigManager(json_name=args.json)

# check if output folder exists
if not os.path.isdir(cfg.output_dir):
    os.makedirs(cfg.output_dir)

# create models
model = Conv1DNet(cfg=cfg)
model.load_state_dict(torch.load(cfg.model.weights, map_location=f'cuda:{cfg.system.gpu}'))

# print model weights into a text file for each layer
with open(os.path.join(cfg.output_dir, 'model_weights.h'), 'w') as f:

    # write header first lines
    f.write('// model_weights.h  (header file for model weights)\n')
    f.write('\n')
    f.write('#ifndef MODEL_WEIGHTS_H\n')
    f.write('#define MODEL_WEIGHTS_H\n')
    f.write('\n')

    batch_norm_layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Printing model weights for layer:')
            print(name)

            # write Eigen Matrix for layer
            layer_name = name.replace('.', '_')
            if len(param.data.shape) == 3:

                # iterate over the second dimension (conv channels)
                num_kernels_variable_name = layer_name + "_num_kernels"
                num_channels_variable_name = layer_name + "_num_channels"
                kernel_size_variable_name = layer_name + "_kernel_size"
                f.write(f"const int {num_kernels_variable_name} = {param.data.size(0)};\n")
                f.write(f"const int {num_channels_variable_name} = {param.data.size(1)};\n")
                f.write(f"const int {kernel_size_variable_name} = {param.data.size(2)};\n")
                f.write(f"const float {layer_name}[{num_kernels_variable_name}][{num_channels_variable_name}][{kernel_size_variable_name}] = ")
                f.write('{ \n')
                for i in range(param.data.size(0)):
                    f.write('   { //')
                    f.write(f' kernel number {i+1}\n')
                    for j in range(param.data.size(1)):
                        f.write('       {')
                        for k in range(param.data.size(2)):
                            f.write("{:.3f}f".format(param.data[i][j][k].item()))
                            if k != param.data.size(2) - 1:
                                f.write(', ')
                            elif j < param.data.size(1) - 1: # end of row
                                f.write('},')
                        if j < param.data.size(1) - 1:
                            f.write('\n')
                        else:
                            f.write('}\n')
                    if i < param.data.size(0) - 1:
                        f.write('   },\n')
                    else:   
                        f.write('   }\n') # end of 2d block
                f.write('};\n\n')

            elif len(param.data.shape) == 2:
                num_rows_variable_name = layer_name + "_num_rows"
                num_cols_variable_name = layer_name + "_num_cols"
                f.write(f"const int {num_rows_variable_name} = {param.data.size(0)};\n")
                f.write(f"const int {num_cols_variable_name} = {param.data.size(1)};\n")
                f.write(f"const float {layer_name}[{num_rows_variable_name}][{num_cols_variable_name}] = ")
                f.write('{ \n')
                for i in range(param.data.size(0)):
                    f.write('    {')
                    for j in range(param.data.size(1)):
                        f.write("{:.3f}f".format(param.data[i][j].item()))
                        if j != param.data.size(1) - 1:
                            f.write(', ')
                        elif i < param.data.size(0) - 1: # end of row
                            f.write('},')
                    if i < param.data.size(0) - 1:
                        f.write('\n')
                    else:
                        f.write('}\n')
                f.write('};\n\n')

            else: # len(param.data.shape) == 1

                # skip batchnorm layer
                if 'bn' in layer_name[:2] or 'fc_layers_1' in layer_name:
                    batch_norm_layer_names.append(layer_name)
                    continue

                size_variable_name = layer_name + "_size"
                f.write(f"const int {size_variable_name} = {param.data.size(0)};\n")
                f.write(f"const float {layer_name}[{size_variable_name}] = ")
                f.write('{ \n')
                f.write('    ')
                for i in range(param.data.size(0)):
                    f.write("{:.3f}f".format(param.data[i].item()))
                    if i < param.data.size(0) - 1:
                        f.write(', ')
                    else:
                        f.write('};\n')
                f.write('\n')

    # add batchnorm layer - manual
    layer_name = 'bn1'
    bn_module = model.bn1
    # weight (or gamma)
    f.write(f"const int {layer_name}_size = {bn_module.weight.size(0)};\n")
    f.write(f"const float {layer_name}_weight[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module.weight.size(0)):
        f.write("{:.3f}f".format(bn_module.weight[i].item()))
        if i < bn_module.weight.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    # bias (or beta)
    f.write(f"const float {layer_name}_bias[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module.bias.size(0)):
        f.write("{:.3f}f".format(bn_module.bias[i].item()))
        if i < bn_module.bias.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    # running mean
    f.write(f"const float {layer_name}_running_mean[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module.running_mean.size(0)):
        f.write("{:.3f}f".format(bn_module.running_mean[i].item()))
        if i < bn_module.running_mean.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')
    
    # running variance
    f.write(f"const float {layer_name}_running_var[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module.running_var.size(0)):
        f.write("{:.3f}f".format(bn_module.running_var[i].item()))
        if i < bn_module.running_var.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    # add second batchnorm layer - manual
    layer_name = 'bn2'
    bn_module2 = model.fc_layers[1]
    # weight (or gamma)
    f.write(f"const int {layer_name}_size = {bn_module2.weight.size(0)};\n")
    f.write(f"const float {layer_name}_weight[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module2.weight.size(0)):
        f.write("{:.3f}f".format(bn_module2.weight[i].item()))
        if i < bn_module2.weight.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    # bias (or beta)
    f.write(f"const float {layer_name}_bias[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module2.bias.size(0)):
        f.write("{:.3f}f".format(bn_module2.bias[i].item()))
        if i < bn_module2.bias.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    # running mean
    f.write(f"const float {layer_name}_running_mean[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module2.running_mean.size(0)):
        f.write("{:.3f}f".format(bn_module2.running_mean[i].item()))
        if i < bn_module2.running_mean.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')
    
    # running variance
    f.write(f"const float {layer_name}_running_var[{layer_name}_size] = ")
    f.write('{ \n')
    f.write('    ')
    for i in range(bn_module2.running_var.size(0)):
        f.write("{:.3f}f".format(bn_module2.running_var[i].item()))
        if i < bn_module2.running_var.size(0) - 1:
            f.write(', ')
        else:
            f.write('};\n')

    f.write('\n')

    # write footer last lines
    f.write('#endif\n')

print('Model weights extracted to path:', os.path.join(cfg.output_dir, 'model_weights.h'))
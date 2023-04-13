#########################################################################
##         This file is part of the alpha-beta-CROWN verifier          ##
##                                                                     ##
## Copyright (C) 2021, Huan Zhang <huan@huan-zhang.com>                ##
##                     Kaidi Xu <xu.kaid@northeastern.edu>             ##
##                     Shiqi Wang <sw3215@columbia.edu>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Yihan Wang <yihanwang@ucla.edu>                 ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from io import FileIO
from torch.nn import functional as F
import torch.nn as nn
from collections import OrderedDict
import math

########################################
# Defined the model architectures
########################################

import numpy as np
import torch
import os

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


ACTIVS = {
    "sigmoid": nn.Sigmoid(),
    "ReLU": nn.ReLU(True),
    "tanh": nn.Tanh(),
    "Affine": None,
    }
class AttitudeController(nn.Module):
    def __init__(self, path = os.path.join(os.path.dirname(__file__), "models/POLAR/AttitudeControl/CLF_controller_layer_num_3_new"), sign = 1):
        super().__init__()
        self.path = path
        self.sign = sign
        self.output_offset = 0.0
        self.output_scale = 1.0
        self.input_size = None
        self.output_size = None
        self.num_layers = None
        self.layers = None

        self.first_weight_mat = None
        self.layer_filters = []

        self.last_weght_mat = None
        self.last_bias_mat = None
        self.unsign_offset_scale = lambda x: x * self.sign

        self.load_from_path(path)
        

    def forward(self, x):
        return self.layers(x)

    def scale(self, w = None, b = None, device = 'cuda'):
        state_dict = self.state_dict()
        print("Scale the first layer with w: {}, b: {}".format(w, b))
        if w is not None:
            weight_mat = np.eye(self.input_size)
            for idx in range(len(w)):
                weight_mat[idx, idx] = w[idx]
            state_dict['layers.lin0.weight'] = torch.tensor(weight_mat.T).to(device)

            #print(self.last_bias_mat)
        if b is not None:
            bias_mat = np.zeros([self.input_size])
            for idx in range(len(b)):
                bias_mat[idx] = b[idx]
            state_dict['layers.lin0.bias'] = torch.tensor(bias_mat.T).to(device)

        if w is None and b is None:
            state_dict['layers.lin0.weight'] = torch.tensor(np.eye(self.input_size).T).to(device)
            state_dict['layers.lin0.bias'] = torch.tensor(np.zeros([self.input_size]).T).to(device)

        self.load_state_dict(state_dict)

    def filter(self, idx = None, device = 'cuda'):
        state_dict = self.state_dict()
        print("Setting filter layer to output channel {}".format(idx))
        if idx is not None:
            weight_mat = np.zeros([self.output_size, self.output_size])
            weight_mat[idx, idx] = 1.
            bias_mat = np.dot(self.last_bias_mat.T, weight_mat.T)

            weight_mat = np.dot(self.last_weight_mat.T, weight_mat.T)
            state_dict['layers.lin{}.weight'.format(self.num_layers + 1)] = torch.tensor(weight_mat.T).to(device)
            #print(self.last_bias_mat)

            state_dict['layers.lin{}.bias'.format(self.num_layers + 1)] = torch.tensor(bias_mat.T).to(device)
        else:
            weight_mat = np.empty_like(self.last_weight_mat)
            weight_mat[:, :] = self.last_weight_mat[:, :]
            # Set specific channel to output
            state_dict['layers.lin{}.weight'.format(self.num_layers + 1)] = torch.tensor(weight_mat.T).to(device)
            bias_mat = np.empty_like(self.last_bias_mat)
            bias_mat[:] = self.last_bias_mat[:]
            state_dict['layers.lin{}.bias'.format(self.num_layers + 1)] = torch.tensor(bias_mat.T).to(device)
        self.load_state_dict(state_dict)


    def load_from_path(self, path = None):
        if path is None:
            path = self.path
        conf_lst = list()
        layers = []
        state_dict = {}
        weight_mat = None
        bias_mat = None
        with open(path, 'r') as f:
            print(">>>>>>>>> Loading Attitude Controller from {}".format(path))
            line = f.readline().split('\n')[0]
            if not line:
                raise FileNotFoundError("No line in the file {}".format(path))
            else:
                self.input_size = int(line)
                print("Number of Inputs: {}".format(self.input_size))
            cnt = 1

            while cnt < 3:
                line = f.readline().split('\n')[0]
                print("Line {}: {}".format(cnt, line.strip()))
                if cnt == 1:
                    self.output_size = int(line)
                    print("Number of Outputs: {}".format(self.output_size))
                    cnt += 1
                    continue
                elif cnt == 2:
                    self.num_layers = int(line)
                    print("Number of Hidden Layers: {}".format(self.num_layers))
                    cnt += 1


            while cnt < 3 + 2 * self.num_layers + 1:
                line = f.readline().split('\n')[0]
                print("Line {}: {}".format(cnt, line))
                cnt += 1
                if(len(layers) < self.num_layers):
                    layers.append(int(line))
                    print("Layer {} Size: {}".format(\
                            len(layers), \
                            layers[-1]))
                else:
                    layers.append(line)
                    print("Activation Function {}: {}".format(\
                            len(layers) - self.num_layers, \
                            layers[-1]))

            print(self.num_layers,  self.output_size, layers)
            if cnt != 3 + 2 * self.num_layers + 1:
                raise ValueError("Line count {} does not match {}".format(cnt, 3 + 2 * self.num_layers))
            
            # layer_tuples = [("lin0", nn.Linear(self.input_size, self.input_size))]
            # print("Added {}".format(layer_tuples[-1]))
            layer_tuples = []

            layer_tuples.append(("lin1", nn.Linear(self.input_size, layers[0])))
            print("Added {}".format(layer_tuples[-1]))
            if layers[self.num_layers] != 'Affine':
                layer_tuples.append((layers[self.num_layers] + "1", ACTIVS[layers[self.num_layers]])),
                print("Added {}".format(layer_tuples[-1]))

            for i in range(self.num_layers - 1):

                layer_tuples.append(
                    (
                        "lin{}".format(i + 2),
                        nn.Linear(
                            layers[i],
                            layers[i + 1])
                    )
                )
                print("Added {}".format(layer_tuples[-1]))

                if layers[i + 1 + self.num_layers] != 'Affine':
                    layer_tuples.append(
                        (
                            layers[i + 1 + self.num_layers] + "{}".format(i + 2),
                            ACTIVS[layers[i + 1 + self.num_layers]]
                        )
                    )
                    print("Added {}".format(layer_tuples[-1]))
                else:
                    print("Not added {}".format(layers[i + 1 + self.num_layers]))


            layer_tuples.append(
                (
                    "lin{}".format(self.num_layers + 1),
                    nn.Linear(
                        layers[self.num_layers - 1],
                        self.output_size)
                )
            )
            print("Added {}".format(layer_tuples[-1]))

            if layers[-1] != 'Affine':
                layer_tuples.append(
                    (
                        layers[-1] + "{}".format(self.num_layers + 1),
                        ACTIVS[layers[-1]]
                    )
                )
                print("Added {}".format(layer_tuples[-1]))
            else:
                print("Not added {}".format(layers[i + 1 + self.num_layers]))

            if self.sign < 0:
                layer_tuples.append(
                    (
                        "lin{}".format(self.num_layers + 2),
                        nn.Linear(
                            self.output_size,
                            self.output_size)
                    )
                )
                print("Added {}".format(layer_tuples[-1]))

            # To select which channel to output, by default the weight should be an identity matrix
            """
            layer_tuples.append(
                (
                    "lin_filter".format(self.num_layers + 2),
                    nn.Linear(
                        self.output_size,
                        self.output_size)
                )
            )
            print("Added {}".format(layer_tuples[-1]))
            """

            self.layers = nn.Sequential(OrderedDict(layer_tuples))

            state_dict = self.state_dict().copy()
            print(state_dict.keys())
            num_layer = 1

            # state_dict['layers.lin0.weight'] = torch.tensor(np.eye(self.input_size).T)
            # state_dict['layers.lin0.bias'] = torch.tensor(np.zeros([self.input_size]).T)
            for i_layer in range(0, len(self.layers) - (1 if self.sign<0 else 0)):
                if not isinstance(self.layers[i_layer], nn.Linear):
                    continue
                layer = self.layers[i_layer]
                print(layer.in_features, layer.out_features)


                bias_mat = np.zeros((layer.out_features))
                 # Set default weights/bias for the last layer then break
                """
                if i_layer == len(self.layers) - 1:
                    weight_mat = np.eye(self.output_size)
                    state_dict['layers.lin_filter.weight'.format(i_layer)] = torch.tensor(weight_mat.T)
                    state_dict['layers.lin_filter.bias'.format(i_layer)] = torch.tensor(bias_mat.T)
                    break
                """

                weight_mat = np.zeros((layer.out_features, layer.in_features))
                for i in range(layer.out_features):
                    for j in range(layer.in_features):
                        line = f.readline().split('\n')[0]
                        weight_mat[i,j] = float(line)
                        cnt += 1
                    line = f.readline().split('\n')[0]
                    bias_mat[i] = float(line)
                    cnt += 1
                # offset = cnt
                # while cnt < offset + weight_mat.shape[0] * weight_mat.shape[1]:
                #     line = f.readline().split('\n')[0]
                #     coord = np.unravel_index(cnt - offset, weight_mat.shape)
                #     np.put(weight_mat, coord, float(line))
                #     cnt += 1


                if num_layer == self.num_layers + 1:
                    self.last_weight_mat = np.empty_like(weight_mat)
                    self.last_weight_mat[:, :] = weight_mat[:, :]
                    # Just for testing
                    #weight_mat_ = np.zeros([self.output_size, self.output_size])
                    #weight_mat_[2, 2] = 1.
                    #weight_mat = np.dot(self.last_weight_mat, weight_mat_)

                state_dict['layers.lin{}.weight'.format(num_layer)] = torch.tensor(weight_mat)
                weight_mat = None

                # offset = cnt
                # while cnt < offset + bias_mat.shape[0]:
                #     line = f.readline().split('\n')[0]
                #     coord = cnt - offset
                #     np.put(bias_mat, coord, float(line))
                #     cnt += 1
                if num_layer == self.num_layers + 1:
                    self.last_bias_mat = np.empty_like(bias_mat)
                    self.last_bias_mat[:] = bias_mat[:]
                    # Just for testing
                    #weight_mat_ = np.zeros([self.output_size, self.output_size])
                    #weight_mat_[2, 2] = 1.
                    #bias_mat = np.dot(self.last_bias_mat, weight_mat_)

                state_dict['layers.lin{}.bias'.format(num_layer)] = torch.tensor(bias_mat)
                bias_mat = None

                num_layer += 1
            
            if self.sign < 0:
                layer = self.layers[-1]
                weight_mat = -np.eye(layer.in_features)
                bias_mat = np.zeros(layer.out_features)
                state_dict['layers.lin{}.weight'.format(num_layer)] = torch.tensor(weight_mat)
                state_dict['layers.lin{}.bias'.format(num_layer)] = torch.tensor(bias_mat)

            print(">>>>>>>>>>>>>>Done loading Attitude Controller")
            for key, value in state_dict.items():
                print(key, value.shape)
            self.load_state_dict(state_dict)


            line = f.readline().split('\n')[0]
            self.output_offset = float(line) 
            line = f.readline().split('\n')[0]
            self.output_scale = float(line)
            
            self.unsign_offset_scale = lambda x: ((x * self.sign - self.output_offset) * self.output_scale)
          
        
            line = f.readline().split('\n')[0]
            while line:
                print(line)
                line = f.readline().split('\n')[0]
            f.close()

ACTIVS = {
    "sigmoid": nn.Sigmoid(),
    "ReLU": nn.ReLU(True),
    "tanh": nn.Tanh(),
    "Affine": None,
    }

class POLARController(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.path = path
        
        self.input_size = None
        self.output_size = None
        self.num_layers = None
        self.layers = None
    
        self.first_weight_mat = None
        self.layer_filters = []

        self.last_weght_mat = None
        self.last_bias_mat = None

        self.load_from_path(path)
        
        self.offset_scale = lambda x: x

    def forward(self, x):
        return self.layers(x)

    def scale(self, w = None, b = None, device = 'cuda'):
        state_dict = self.state_dict()
        print("Scale the first layer with w: {}, b: {}".format(w, b))
        if w is not None:
            weight_mat = np.eye(self.input_size)
            for idx in range(len(w)):
                weight_mat[idx, idx] = w[idx]
            state_dict['layers.lin0.weight'] = torch.tensor(weight_mat.T).to(device)

            #print(self.last_bias_mat)
        if b is not None:
            bias_mat = np.zeros([self.input_size])
            for idx in range(len(b)):
                bias_mat[idx] = b[idx]
            state_dict['layers.lin0.bias'] = torch.tensor(bias_mat.T).to(device)

        if w is None and b is None:
            state_dict['layers.lin0.weight'] = torch.tensor(np.eye(self.input_size).T).to(device)
            state_dict['layers.lin0.bias'] = torch.tensor(np.zeros([self.input_size]).T).to(device)

        self.load_state_dict(state_dict)

    def filter(self, idx = None, device = 'cuda'):
        state_dict = self.state_dict()
        print("Setting filter layer to output channel {}".format(idx))
        if idx is not None:
            weight_mat = np.zeros([self.output_size, self.output_size])
            weight_mat[idx, idx] = 1.
            bias_mat = np.dot(self.last_bias_mat, weight_mat)

            weight_mat = np.dot(self.last_weight_mat, weight_mat)
            state_dict['layers.lin{}.weight'.format(self.num_layers + 1)] = torch.tensor(weight_mat.T).to(device)
            #print(self.last_bias_mat)

            state_dict['layers.lin{}.bias'.format(self.num_layers + 1)] = torch.tensor(bias_mat.T).to(device)
        else:
            weight_mat = np.empty_like(self.last_weight_mat)
            weight_mat[:, :] = self.last_weight_mat[:, :]
            # Set specific channel to output
            state_dict['layers.lin{}.weight'.format(self.num_layers + 1)] = torch.tensor(weight_mat.T).to(device)
            bias_mat = np.empty_like(self.last_bias_mat)
            bias_mat[:] = self.last_bias_mat[:]
            state_dict['layers.lin{}.bias'.format(self.num_layers + 1)] = torch.tensor(bias_mat.T).to(device)
        self.load_state_dict(state_dict)
    
    def negate(self, device = 'cuda'):
        state_dict = self.state_dict()
        state_dict['layers.lin{}.weight'.format(self.num_layers + 2)] = state_dict['layers.lin{}.weight'.format(self.num_layers + 2)] * -1.0
        self.load_state_dict(state_dict)
      


    def load_from_path(self, path = None):
        if path is None:
            path = self.path
        conf_lst = list();
        layers = []
        state_dict = {}
        weight_mat = None
        bias_mat = None
        with open(path, 'r') as f:
            print(">>>>>>>>> Loading model from {}".format(path))
            line = f.readline().split('\n')[0]
            if not line:
                raise FileNotFoundError("No line in the file {}".format(path))
            else:
                self.input_size = int(line)
                print("Number of Inputs: {}".format(self.input_size))
            cnt = 1

            while cnt < 3:
                line = f.readline().split('\n')[0]
                print("Line {}: {}".format(cnt, line.strip()))
                if cnt == 1:
                    self.output_size = int(line)
                    print("Number of Outputs: {}".format(self.output_size))
                    cnt += 1
                    continue
                elif cnt == 2:
                    self.num_layers = int(line)
                    print("Number of Hidden Layers: {}".format(self.num_layers))
                    cnt += 1


            while cnt < 3 + 2 * self.num_layers + 1:
                line = f.readline().split('\n')[0]
                print("Line {}: {}".format(cnt, line))
                cnt += 1
                if(len(layers) < self.num_layers):
                    layers.append(int(line))
                    print("Layer {} Size: {}".format(\
                            len(layers), \
                            layers[-1]))
                else:
                    layers.append(line)
                    print("Activation Function {}: {}".format(\
                            len(layers) - self.num_layers, \
                            layers[-1]))

            print(self.num_layers,  self.output_size, layers)
            if cnt != 3 + 2 * self.num_layers + 1:
                raise ValueError("Line count {} does not match {}".format(cnt, 3 + 2 * self.num_layers))
            else:
                layer_tuples = [("lin0", nn.Linear(self.input_size, self.input_size))]
                print("Added {}".format(layer_tuples[-1]))

                layer_tuples.append(("lin1", nn.Linear(self.input_size, layers[0])))
                print("Added {}".format(layer_tuples[-1]))
                if layers[self.num_layers] != "Affine":
                    layer_tuples.append((layers[self.num_layers] + "1", ACTIVS[layers[self.num_layers]])),
                    print("Added {}".format(layer_tuples[-1]))

                for i in range(self.num_layers - 1):

                    layer_tuples.append(
                        (
                            "lin{}".format(i + 2),
                            nn.Linear(
                                layers[i],
                                layers[i + 1])
                        )
                    )
                    print("Added {}".format(layer_tuples[-1]))

                    if layers[i + 1 + self.num_layers] != "Affine":
                        layer_tuples.append(
                            (
                                layers[i + 1 + self.num_layers] + "{}".format(i + 2),
                                ACTIVS[layers[i + 1 + self.num_layers]]
                            )
                        )
                        print("Added {}".format(layer_tuples[-1]))
                    else:
                        print("Not added {}".format(layers[i + 1 + self.num_layers]))


                layer_tuples.append(
                    (
                        "lin{}".format(self.num_layers + 1),
                        nn.Linear(
                            layers[self.num_layers - 1],
                            self.output_size)
                    )
                )
                print("Added {}".format(layer_tuples[-1]))

                if layers[-1] != "Affine":
                    layer_tuples.append(
                        (
                            layers[i + 1 + self.num_layers],
                            ACTIVS[layers[-1]]
                        )
                    )
                    print("Added {}".format(layer_tuples[-1]))
                else:
                    print("Not added {}".format(layers[-1]))


                # To select which channel to output, by default the weight should be an identity matrix
                 
                layer_tuples.append(
                    (
                        "lin{}".format(self.num_layers + 2),
                        nn.Linear(
                            self.output_size,
                            self.output_size)
                    )
                )
                print("Added {}".format(layer_tuples[-1]))
                 

                self.layers = nn.Sequential(OrderedDict(layer_tuples))

            state_dict = self.state_dict().copy()
            print(state_dict.keys())
            num_layer = 1

            state_dict['layers.lin0.weight'] = torch.tensor(np.eye(self.input_size).T)
            state_dict['layers.lin0.bias'] = torch.tensor(np.zeros([self.input_size]).T)
            for i_layer in range(1, len(self.layers)):
                if not isinstance(self.layers[i_layer], nn.Linear):
                    continue
                layer = self.layers[i_layer]
                print(layer.in_features, layer.out_features)


                bias_mat = np.zeros((layer.out_features))
                 # Set default weights/bias for the last layer then break
                """
                if i_layer == len(self.layers) - 1:
                    weight_mat = np.eye(self.output_size)
                    state_dict['layers.lin_filter.weight'.format(i_layer)] = torch.tensor(weight_mat.T)
                    state_dict['layers.lin_filter.bias'.format(i_layer)] = torch.tensor(bias_mat.T)
                    break
                """
                if num_layer == self.num_layers + 2:
                    weight_mat = np.eye(self.output_size)
                else:
                    weight_mat = np.zeros((layer.in_features, layer.out_features))
                    offset = cnt
                    while cnt < offset + weight_mat.shape[0] * weight_mat.shape[1]:
                        line = f.readline().split('\n')[0]
                        coord = np.unravel_index(cnt - offset, weight_mat.shape)
                        np.put(weight_mat, coord, float(line))
                        cnt += 1
                    if num_layer == self.num_layers + 1:
                        self.last_weight_mat = np.empty_like(weight_mat)
                        self.last_weight_mat[:, :] = weight_mat[:, :]
                        # Just for testing
                        #weight_mat_ = np.zeros([self.output_size, self.output_size])
                        #weight_mat_[2, 2] = 1.
                        #weight_mat = np.dot(self.last_weight_mat, weight_mat_)
                    offset = cnt
                    while cnt < offset + bias_mat.shape[0]:
                        line = f.readline().split('\n')[0]
                        coord = cnt - offset
                        np.put(bias_mat, coord, float(line))
                        cnt += 1
                    if num_layer == self.num_layers + 1:
                        self.last_bias_mat = np.empty_like(bias_mat)
                        self.last_bias_mat[:] = bias_mat[:]
                        # Just for testing
                        #weight_mat_ = np.zeros([self.output_size, self.output_size])
                        #weight_mat_[2, 2] = 1.
                        #bias_mat = np.dot(self.last_bias_mat, weight_mat_)
                
                state_dict['layers.lin{}.weight'.format(num_layer)] = torch.tensor(weight_mat.T)
                weight_mat = None
                state_dict['layers.lin{}.bias'.format(num_layer)] = torch.tensor(bias_mat.T)
                bias_mat = None

                num_layer += 1

            self.load_state_dict(state_dict)

            
            line_offset = f.readline().split('\n')[0] * self.sign
            line_scale = f.readline().split('\n')[0] * self.sign
            print("Offset: {}       Scale: {}".format(line_offset, line_scale))
            print(">>>>>>>>>>>>>>Done loading Attitude Controller")
            
            self.offset_scale = lambda x: (x - float(line_offset)) * (float(line_scale))
            f.close()


if __name__ == "__main__":
    nn = AttitudeController()

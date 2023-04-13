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

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            # can do planes 32
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        # print("residual relu:", out.shape, out[0].view(-1).shape)
        return out


class CResNet5(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet5, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out


class CResNet7(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet7, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out


def resnet4b():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")

def resnet2b():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet5_16_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="dense")

def cresnet5_16_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="avg")


def cresnet5_8_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet5_8_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet5_4_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet5_4_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")


def cresnet7_8_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet7_8_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet7_4_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet7_4_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")


def cresnet5_16_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")


def cresnet5_16_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="avg")


def cresnet5_8_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet5_8_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet5_4_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet5_4_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


def cresnet7_8_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet7_8_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet7_4_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet7_4_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x, W in zip(xs, self.Ws) if W is not None)
        return out


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]


def model_resnet(in_ch=3, in_dim=32, width=1, mult=16, N=1):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)),
            nn.ReLU(),
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0),
                  None,
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)),
            nn.ReLU()
        ]

    conv1 = [nn.Conv2d(in_ch, mult, 3, stride=1, padding=3 if in_dim == 28 else 1), nn.ReLU()]
    conv2 = block(mult, mult * width, 3, False)
    for _ in range(N):
        conv2.extend(block(mult * width, mult * width, 3, False))
    conv3 = block(mult * width, mult * 2 * width, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(mult * 2 * width, mult * 2 * width, 3, False))
    conv4 = block(mult * 2 * width, mult * 4 * width, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(mult * 4 * width, mult * 4 * width, 3, False))
    layers = (
            conv1 +
            conv2 +
            conv3 +
            conv4 +
            [Flatten(),
             nn.Linear(mult * 4 * width * 8 * 8, 1000),
             nn.ReLU(),
             nn.Linear(1000, 10)]
    )
    model = DenseSequential(
        *layers
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def mnist_fc():
    # cifar base
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    )
    return model


def cifar_model_base():
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    # cifar deep
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cnn_4layer():
    # cifar_cnn_a
    return cifar_model_wide()


def cnn_4layer_adv():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_adv4():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_mix4():
    # cifar_cnn_a_mix4
    return cifar_model_wide()


def cnn_4layer_b():
    # cifar_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cnn_4layer_b4():
    # cifar_cnn_b4
    return cnn_4layer_b()

def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def cifar_conv_small():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv_big():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_conv_small():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*5*5,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_big():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_6_100():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100, 10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_9_100():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_6_200():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_9_200():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_fc1():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    return model


def mnist_fc2():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fc3():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fc_3_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_4_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_5_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_6_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_7_512():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_madry_secret():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    return model


def cifar_conv1():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv3():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2048, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv4():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv5():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv6():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def MadryCNN():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_one_maxpool():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_no_maxpool():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4, 8, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


def MadryCNN_one_maxpool_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


def MadryCNN_no_maxpool_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


class TradesCNN(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class TradesCNN_one_maxpool(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, stride=2)),
            ('relu2', activ),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class TradesCNN_no_maxpool(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, stride=2)),
            ('relu2', activ),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3, stride=2)),
            ('relu4', activ),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

import numpy as np
import torch
import os
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
        self.offset_scale = lambda x: x * self.sign

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
                        layers[i + 1 + self.num_layers],
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
            print(">>>>>>>>>> Offset: {}  >>>>>> Scale: {}>>>>>>>".format(self.output_offset, self.output_scale))
            
            self.unsign_offset_scale = lambda x: (x * self.sign - self.output_offset) * self.output_scale 

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

            
            line_offset = f.readline().split('\n')[0]
            line_scale = f.readline().split('\n')[0]
            print("Offset: {}       Scale: {}".format(line_offset, line_scale))
            print(">>>>>>>>>>>>>>Done loading Attitude Controller")

            line = f.readline().split('\n')[0]
            self.output_offset = float(line)
            line = f.readline().split('\n')[0]
            self.output_scale = float(line)
            print(">>>>>>>>>> Offset: {}  >>>>>> Scale: {}>>>>>>>".format(self.output_offset, self.output_scale))
            
            self.offset_scale = lambda x: (x - self.output_offset) * self.output_scale

            
            f.close()


if __name__ == "__main__":
    nn = AttitudeController()

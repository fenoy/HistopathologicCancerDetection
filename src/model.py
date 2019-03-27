from torch import nn
import torchvision
import numpy as np
import random
import torch

class SELayer(nn.Module):
    def __init__(self, n_channels, reduction):
        super(SELayer, self).__init__()

        self.global_average = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.global_average(x).view(b, -1)
        scale = self.se(scale).view(b, c, 1, 1)
        return scale.expand_as(x) * x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, hidden_planes, stride=1, groups=32):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, 1, 0, bias=False, groups=groups),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_planes, hidden_planes, 3, stride, 1, bias=False, groups=groups),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_planes, 2 * hidden_planes, 1, 1, 0, bias=False, groups=groups),
            nn.BatchNorm2d(2 * hidden_planes)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_planes, 1, stride, 0, bias=False),
            nn.BatchNorm2d(2 * hidden_planes)
        )

        self.se = SELayer(2 * hidden_planes, 16)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.bottleneck(x)
        residual = self.se(residual)
        x = self.downsample(x)
        return self.relu(residual + x)


class Block(nn.Module):
    def __init__(self, n, in_planes, hidden_planes):
        super(Block, self).__init__()

        layers = [Bottleneck(2 * hidden_planes, hidden_planes) for _ in range(n - 1)]

        self.block = nn.Sequential(
            Bottleneck(in_planes, hidden_planes, stride=2),
            *layers
        )

    def forward(self, x):
        return self.block(x)


class SE_ResNet(nn.Module):
    def __init__(self, mult, num_classes):
        super(ResNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, mult, 5, 2, 2),
            nn.BatchNorm2d(mult),
            nn.ReLU(inplace=True),

            Block(3, mult, 2 * mult),
            Block(4, 4 * mult, 4 * mult),
            Block(6, 8 * mult, 8 * mult),
            Block(3, 16 * mult, 16 * mult),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * mult, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, mult, num_classes):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, mult, 5, 2, 2),
            nn.BatchNorm2d(mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(mult, 2 * mult, 5, 2, 2),
            nn.BatchNorm2d(2 * mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * mult, 4 * mult, 3, 2, 1),
            nn.BatchNorm2d(4 * mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(4 * mult, 8 * mult, 3, 2, 1),
            nn.BatchNorm2d(8 * mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(8 * mult, 16 * mult, 3, 2, 1),
            nn.BatchNorm2d(16 * mult),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * mult, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

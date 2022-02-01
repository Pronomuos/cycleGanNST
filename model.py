import torch.nn as nn
from torch.nn import functional as F


class ResnetGenerator(nn.Module):

    def __init__(self, n_input_channels=3, n_filters=64, n_output_channels=3, use_dropout=False):
        super(ResnetGenerator, self).__init__()

        # down_sampling

        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_input_channels, n_filters, kernel_size=7),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU()
        )
        self.down_sampling1 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(n_filters * 2),
            nn.ReLU()
        )
        self.down_sampling2 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(n_filters * 2),
            nn.ReLU()
        )

        # residual blocks

        self.residual_blks = []
        for _ in range(9):
            self.residual_blks += [Residual(n_filters * 4, use_dropout)]
        self.residual_blks = nn.Sequential(*self.residual_blks)

        # up_sampling

        self.up_sampling1 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(n_filters * 2),
            nn.ReLU()
        )
        self.up_sampling2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, n_output_channels, kernel_size=7),
            nn.InstanceNorm2d(n_output_channels),
            nn.Tanh()
        )

    def forward(self, X):
        X = self.initial_conv(X)
        X = self.down_sampling1(X)
        X = self.down_sampling2(X)

        X = self.residual_blks(X)

        X = self.up_sampling1(X)
        X = self.up_sampling2(X)
        X = self.final_conv(X)

        return X


class Residual(nn.Module):

    def __init__(self, n_channels, use_dropout=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels,
                               kernel_size=3, padding=1)
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.bn = nn.InstanceNorm2d(n_channels)

    def forward(self, X):
        Y = F.relu(self.bn(self.conv1(X)))
        if self.dropout:
            Y = self.dropout(Y)
        Y = self.bn(self.conv2(Y))
        Y += X
        return F.relu(Y)


class PatchGanDiscriminator(nn.Module):

    def __init__(self, n_input_channels=3, n_filters=64):
        super(PatchGanDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, n_filters, kernel_size=4, stride=2, padding=1)
        self.bn = nn.InstanceNorm2d(n_filters)
        self.main = []
        for _ in range(3):
            prev_channels = n_filters
            n_filters *= 2
            self.main += [nn.Conv2d(prev_channels, n_filters, kernel_size=4, stride=2, padding=1),
                          nn.InstanceNorm2d(n_filters),
                          nn.LeakyReLU()]

        self.main += [nn.Conv2d(n_filters, n_filters, kernel_size=4, stride=2, padding=1),
                      nn.InstanceNorm2d(n_filters),
                      nn.LeakyReLU()]
        self.main = nn.Sequential(*self.main)

        self.conv2 = nn.Conv2d(n_filters, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, X):
        X = F.leaky_relu_(self.bn(self.conv1(X)))
        X = self.main(X)

        return self.conv2(X)



import numpy as np
import torch
import torch.nn as nn


class EncoderMiniBlock(nn.Module):
    def __init__(self, in_filters, n_filters=32, dropout_prob=0.3, max_pooling=True):
        super(EncoderMiniBlock, self).__init__()

        self.max_pool = None
        if max_pooling:
            self.max_pool = nn.MaxPool2d(2)

        self.conv1=nn.Conv2d(in_filters, n_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.batchnorm1=nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

        self.drop = None
        if dropout_prob > 0:
            self.drop = nn.Dropout(dropout_prob)


    def forward(self, x):
        if self.max_pool:
            x=self.max_pool(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        # print(x.detach().numpy()[0, 0, 10,: 10])

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        if self.drop:
            x = self.drop(x)
        return x

class DecoderMiniBlock(nn.Module):
    def __init__(self, in_filters=32, n_filters=32):
        super(DecoderMiniBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_filters, in_filters // 2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.batchnorm1 = nn.BatchNorm2d(in_filters // 2, momentum=0.99, eps=0.001)

        self.conv2 = nn.Conv2d(in_filters // 2, n_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.batchnorm2 = nn.BatchNorm2d(n_filters, momentum=0.99, eps=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        return x

class unet(nn.Module):
    def __init__(self, n_channels, n_classes, a=1):
        super(unet, self).__init__()

        n_filters = int(n_channels * a)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cblock1 = EncoderMiniBlock(4, n_filters, dropout_prob=0, max_pooling=False)
        self.cblock2 = EncoderMiniBlock(n_filters, n_filters * 2, dropout_prob=0, max_pooling=True)
        self.cblock3 = EncoderMiniBlock(n_filters * 2, n_filters * 4, dropout_prob=0, max_pooling=True)
        self.cblock4 = EncoderMiniBlock(n_filters * 4, n_filters * 8, dropout_prob=0.3, max_pooling=True)
        self.cblock5 = EncoderMiniBlock(n_filters * 8, n_filters * 8, dropout_prob=0.3, max_pooling=True)
        self.ublock4 = DecoderMiniBlock(n_filters * 8 * 2, n_filters * 8 // 2)
        self.ublock3 = DecoderMiniBlock(n_filters * 4 * 2, n_filters * 4 // 2)
        self.ublock2 = DecoderMiniBlock(n_filters * 2 * 2, n_filters)
        self.ublock1 = DecoderMiniBlock(n_filters * 2, n_filters)

        self.out_conv = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        c1 = self.cblock1(x)
        c2 = self.cblock2(c1)
        c3 = self.cblock3(c2)
        c4 = self.cblock4(c3)
        c5 = self.cblock5(c4)

        # upsample previous, then concatenate
        c5 = self.up(c5)
        merge = torch.cat([c5, c4], dim=1)
        d = self.ublock4(merge)

        d = self.up(d)
        merge = torch.cat([d, c3], dim=1)
        d = self.ublock3(merge)

        d = self.up(d)
        merge = torch.cat([d, c2], dim=1)
        d = self.ublock2(merge)

        d = self.up(d)
        merge = torch.cat([d, c1], dim=1)
        d = self.ublock1(merge)

        return self.sig(self.out_conv(d))


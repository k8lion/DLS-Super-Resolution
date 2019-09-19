
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


"""
PyTorch implementation of:
[1] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
With a few modifications as suggested by:
[2] [Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)
"""


def create_loss_model(vgg, end_layer, use_maxpool=True, use_cuda=False):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)

    model = nn.Sequential()

    if use_cuda:
        model.cuda(device_id=0)

    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


class ResidualBlock(nn.Module):
    """ Residual blocks as implemented in [1] """
    def __init__(self, num, use_cuda=False):
        super(ResidualBlock, self).__init__()
        if use_cuda:
            self.c1 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1).cuda(device_id=0)
            self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1).cuda(device_id=0)
            self.b1 = nn.BatchNorm2d(num).cuda(device_id=0)
            self.b2 = nn.BatchNorm2d(num).cuda(device_id=0)
        else:
            self.c1 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
            self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
            self.b1 = nn.BatchNorm2d(num)
            self.b2 = nn.BatchNorm2d(num)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x


class UpsampleBlock(nn.Module):
    """ Upsample block suggested by [2] to remove checkerboard pattern from images """
    def __init__(self, num, use_cuda=False):
        super(UpsampleBlock, self).__init__()
        if use_cuda:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2).cuda(device_id=0)
            self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=0).cuda(device_id=0)
            self.b3 = nn.BatchNorm2d(num).cuda(device_id=0)
        else:
            self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
            self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=0)
            self.b3 = nn.BatchNorm2d(num)

    def forward(self, x):
        h = self.up1(x)
        h = F.pad(h, (1, 1, 1, 1), mode='reflect')
        h = self.b3(self.c2(h))
        return F.relu(h)


class SuperRes4x(nn.Module):
    def __init__(self, use_cuda=False, use_UpBlock=True):

        super(SuperRes4x, self).__init__()
        # To-do: Retrain with self.uplock and self.use_cuda as parameters

        # self.upblock = use_UpBlock
        # self.use_cuda = use_cuda
        upblock = True

        # Downsizing layer
        if use_cuda:
            self.c1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4).cuda(device_id=0)
            self.b2 = nn.BatchNorm2d(64).cuda(device_id=0)
        else:
            self.c1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
            self.b2 = nn.BatchNorm2d(64)

        if upblock:
            # Loop for residual blocks
            self.rs = [ResidualBlock(64, use_cuda=use_cuda) for i in range(4)]
            # Loop for upsampling
            self.up = [UpsampleBlock(64, use_cuda=use_cuda) for i in range(2)]
        else:
            # Loop for residual blocks
            self.rs = [ResidualBlock(64, use_cuda=use_cuda) for i in range(4)]
            # Transposed convolution blocks
            if self.use_cuda:
                self.dc2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1).cuda(device_id=0)
                self.bc2 = nn.BatchNorm2d(64).cuda(device_id=0)
                self.dc3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1).cuda(device_id=0)
                self.bc3 = nn.BatchNorm2d(64).cuda(device_id=0)
            else:
                self.dc2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
                self.bc2 = nn.BatchNorm2d(64)
                self.dc3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
                self.bc3 = nn.BatchNorm2d(64)

        # Last convolutional layer
        if use_cuda:
            self.c3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4).cuda(device_id=0)
        else:
            self.c3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        upblock = True
        # Downsizing layer - Large Kernel ensures large receptive field on the residual blocks
        h = F.relu(self.b2(self.c1(torch.tensor(x).float())))

        # Residual Layers
        for r in self.rs:
            h = r(h)  # will go through all residual blocks in this loop

        if upblock:
            # Upsampling Layers - improvement suggested by [2] to remove "checkerboard pattern"
            for u in self.up:
                h = u(h)  # will go through all upsampling blocks in this loop
        else:
            # As recommended by [1]
            h = F.relu(self.bc2(self.dc2(h)))
            h = F.relu(self.bc3(self.dc3(h)))

        # Last layer and scaled tanh activation - Scaled from 0 to 1 instead of 0 - 255
        h = F.tanh(self.c3(h))
        h = torch.add(h, 1.)
        h = torch.mul(h, 0.5*255)
        return transforms.ToPILImage()(h.select(0,0))

class SuperResolver(torch.nn.Module):
    def __init__(self):
        super(SuperResolver, self).__init__()
        self.itr = 0
        self.superresolver = nn.Sequential(
            nn.ConvTranspose2d(3,3,kernel_size=11, stride=2, padding=5, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(3,3,kernel_size=15, stride=2, padding=7, output_padding = 1),
            nn.Tanh()
        )

    def forward(self, img):
        t = transforms.ToTensor()(img).unsqueeze(0)
        x = self.superresolver(t)
        return transforms.ToPILImage()(x.select(0,0))


class SuperResST(torch.nn.Module):
    def __init__(self):
        super(SuperResST, self).__init__()
        self.itr = 0
        self.d_block3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
        )
        self.residual_in = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1, 1), stride=(1, 1))


        # 512 * 128 * 128
        self.d_up_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 512 * 256 * 256
        self.d_up_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(2, 2), stride=(2, 2))
        )

        # 64 * 256 * 256
        self.d_block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 3 * 256 * 256
        self.d_block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

        # one x one convultion to match shape of up_conv3 and block4
        self.residual_out = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):
        x = transforms.ToTensor()(x).unsqueeze(0)
        a = self.d_block3(x)
        b = self.residual_in(x)
        x = a + b
        x = self.d_up_conv2(x)
        x = self.d_up_conv3(x)
        x = self.d_block4(x) + self.residual_out(x)
        x = self.d_block5(x)

        return transforms.ToPILImage()(x.select(0,0))

class SuperResST8(torch.nn.Module):
    def __init__(self):
        super(SuperResST8, self).__init__()
        self.itr = 0
        self.d_block3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
        )
        self.residual_in = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1, 1), stride=(1, 1))


        # 512 * 128 * 128
        self.d_up_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 512 * 256 * 256
        self.d_up_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 512 * 256 * 256
        self.d_up_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 64 * 256 * 256
        self.d_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 3 * 256 * 256
        self.d_block5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
        )

        # one x one convultion to match shape of up_conv3 and block4
        self.residual_out = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1))

        self.down_sample = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2)),
            nn.Tanh()
        )


    def forward(self, x):
        x = transforms.ToTensor()(x).unsqueeze(0)
        a = self.d_block3(x)
        b = self.residual_in(x)
        x = a + b
        x = self.d_up_conv2(x)
        x = self.d_up_conv3(x)
        x = self.d_up_conv4(x)
        x = self.d_block4(x) + self.residual_out(x)
        x = self.d_block5(x)
        x = self.down_sample(x)
        return transforms.ToPILImage()(x.select(0,0))


import math

import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.itr = 0
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        x = transforms.ToTensor()(x).unsqueeze(0)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        x =  (nn.Tanh()(block8) + 1) / 2
        return transforms.ToPILImage()(x.select(0,0))



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
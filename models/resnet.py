import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



import actquant
from uniq import UNIQNet

__all__ = ['resnet']


class Conv2d(nn.Conv2d):
    """ Create registered buffer for layer base for quantization"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.register_buffer('layer_b', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('initial_clamp_value', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('layer_basis', torch.ones(1))  # Attempt to enable multi-GPU
        # self.layer_basis = Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):

        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        return output


class Linear(nn.Linear):
    """ Create registered buffer for layer base for quantization"""

    def __init__(self, in_features, out_features, bias=True):

        super(Linear, self).__init__(in_features, out_features, bias=True)
        self.register_buffer('layer_b', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('initial_clamp_value', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('layer_basis', torch.ones(1))  # Attempt to enable multi-GPU
        # self.layer_basis = Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):

        output = F.linear(input, self.weight, self.bias)

        return output

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"

    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_()  # The original initialization in class _BatchNorm
            m.bias.data.zero_()  # The original initialization in class _BatchNorm

        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            m.weight.data.normal_(0, math.sqrt(2. / n))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 act_bitwidth=32, act_quant=False, act_noise=False,):
        super(BasicBlock, self).__init__()
        if isinstance(act_bitwidth, list):
            assert (len(act_bitwidth) == 2)
            self.act_bitwidth = act_bitwidth
        else:
            self.act_bitwidth = [act_bitwidth] * 2
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.relu1 = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth[0])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        self.relu2 = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth[1])
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_bitwidth=32, act_quant=False):
        super(Bottleneck, self).__init__()
        if isinstance(act_bitwidth, list):
            assert (len(act_bitwidth) == 3)
            self.act_bitwidth = act_bitwidth
        else:
            self.act_bitwidth = [act_bitwidth] * 3

        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.relu1 = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth[0])
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=32)

        self.bn2 = nn.BatchNorm2d(planes)

        self.relu2 = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth[1])
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu3 = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth[2])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(UNIQNet):
    def __init__(self, quant_epoch_step,quant_start_stage, quant=False, noise=False, bitwidth=32, step=2,
                 quant_edges=True,
                 step_setup=[15, 9], act_bitwidth=32, act_quant=False, noise_mask=0.05):
        super(ResNet, self).__init__(quant_epoch_step=quant_epoch_step, quant_start_stage=quant_start_stage,
                                     quant=quant, noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                    step_setup=step_setup, act_bitwidth=act_bitwidth,
                                     act_quant=act_quant, noise_mask=noise_mask)

    def _make_layer(self, block, planes, blocks, stride=1, act_bitwidth=32, act_quant=False):
        multiplier = 2 if block is BasicBlock else 3
        if isinstance(act_bitwidth, list):
            assert (len(act_bitwidth) == blocks * multiplier)
            act_bitwidth = act_bitwidth
        else:
            act_bitwidth = [act_bitwidth for _ in range(blocks * multiplier)]

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, act_bitwidth=act_bitwidth[0:multiplier],
                            act_quant=act_quant))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, act_bitwidth=act_bitwidth[i * multiplier:(i + 1) * multiplier],
                      act_quant=act_quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_imagenet(ResNet):
    def __init__(self, quant_epoch_step, quant_start_stage, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3],
                 quant=False, noise=False, bitwidth=32, step=2, quant_edges=True,step_setup=[15, 9],
                 act_bitwidth=32, act_quant=False, uniq=True, normalize=False, noise_mask=0.05):
        super(ResNet_imagenet, self).__init__(quant_epoch_step=quant_epoch_step, quant_start_stage=quant_start_stage,
                                              quant=quant, noise=noise, bitwidth=bitwidth, step=step,
                                              quant_edges=quant_edges, step_setup=step_setup,
                                              act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              noise_mask=noise_mask)

        self.inplanes = 64
        self.preprocess = lambda x: x

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        block_size = 2 if block is BasicBlock else 3

        self.relu = actquant.ActQuantBuffers(quant=act_quant, bitwidth=self.act_bitwidth)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        act_nums = [1]
        for i in range(4):
            act_nums.append(act_nums[i] + layers[i] * block_size)
        self.layer1 = self._make_layer(block, 64, layers[0], act_bitwidth=self.act_bitwidth, act_quant=act_quant)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       act_bitwidth=self.act_bitwidth, act_quant=act_quant)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       act_bitwidth=self.act_bitwidth, act_quant=act_quant)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act_bitwidth=self.act_bitwidth,
                                       act_quant=act_quant)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = Linear(512 * block.expansion, num_classes)

        init_model(self)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 5e-2,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 5e-3},
            {'epoch': 60, 'lr': 5e-4, 'weight_decay': 0},
            {'epoch': 90, 'lr': 5e-5}
        ]

class ResNet_cifar10(ResNet):
    def __init__(self, quant_epoch_step, quant_start_stage, num_classes=10, block=BasicBlock, depth=18, quant=False,
                 noise=False, bitwidth=32, step=2, quant_edges=True, act_noise=True, step_setup=[15, 9],
                 act_bitwidth=32, act_quant=False, layers=[2, 2, 2, 2], normalize=False,
                 noise_mask=0.05):
        super(ResNet_cifar10, self).__init__(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage,
                                             quant=quant, noise=noise, bitwidth=bitwidth, step=step,
                                             quant_edges=quant_edges, step_setup=step_setup,
                                             act_bitwidth=act_bitwidth, act_quant=act_quant,
                                             noise_mask=noise_mask)

        self.preprocess = lambda x: x

        self.inplanes = 64
        n = int((depth - 2) / 6)
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        if isinstance(act_bitwidth, list):
            if self.quant_edges:
                assert (len(act_bitwidth) == depth - 1)
            else:
                assert (len(act_bitwidth) == depth - 2)
            self.act_bitwidth = act_bitwidth
        else:
            if self.quant_edges:
                self.act_bitwidth = [act_bitwidth] * (depth - 1)
            else:
                self.act_bitwidth = [act_bitwidth] * (depth - 2)

        self.relu = actquant.ActQuantBuffers(quant=act_quant, bitwidth=act_bitwidth)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 64, layers[0], act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, act_bitwidth=act_bitwidth, act_quant=act_quant)

        # self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(4)
        self.fc = Linear(512, num_classes)


        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]

        # self.prepare_uniq()


def resnet(**kwargs):
    num_classes, depth, dataset, batch_norm, shortcut, quantize, noise, bitwidth, step, act_bitwidth, \
    act_quant, quant_edges, step_setup , quant_epoch_step, quant_start_stage,\
    normalize, noise_mask = map(kwargs.get,
                                                        ['num_classes', 'depth', 'dataset', 'batch_norm', 'shortcut',
                                                         'quantize', 'noise', 'bitwidth', 'step',
                                                         'act_bitwidth', 'act_quant', 'quant_edges',
                                                         'step_setup' , 'quant_epoch_step','quant_start_stage',
                                                         'normalize', 'noise_mask'])
    dataset = dataset or 'imagenet'

    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(quant_epoch_step=quant_epoch_step, quant_start_stage=quant_start_stage, num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2], quant=quantize,
                                   noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                   step_setup=step_setup, act_bitwidth=act_bitwidth,
                                   act_quant=act_quant, normalize=normalize, noise_mask=noise_mask)
        if depth == 34:
            return ResNet_imagenet(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage, num_classes=num_classes, block=BasicBlock, layers=[3, 4, 6, 3], quant=quantize,
                                   noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                   step_setup=step_setup, act_bitwidth=act_bitwidth,
                                   act_quant=act_quant, normalize=normalize, noise_mask=noise_mask)
        if depth == 50:
            return ResNet_imagenet(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage , num_classes=num_classes, block=Bottleneck, layers=[3, 4, 6, 3], quant=quantize,
                                   noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                   step_setup=step_setup, act_bitwidth=act_bitwidth,
                                   act_quant=act_quant, normalize=normalize, noise_mask=noise_mask)
        if depth == 101:
            return ResNet_imagenet(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage, num_classes=num_classes, block=Bottleneck, layers=[3, 4, 23, 3], quant=quantize,
                                   noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                   step_setup=step_setup, act_bitwidth=act_bitwidth,
                                   act_quant=act_quant, normalize=normalize, noise_mask=noise_mask)
        if depth == 152:
            return ResNet_imagenet(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage, num_classes=num_classes, block=Bottleneck, layers=[3, 8, 36, 3], quant=quantize,
                                   noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                   step_setup=step_setup, act_bitwidth=act_bitwidth,
                                   act_quant=act_quant, normalize=normalize, noise_mask=noise_mask)

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 56
        return ResNet_cifar10(quant_epoch_step=quant_epoch_step, quant_start_stage=quant_start_stage,
                              num_classes=num_classes, block=BasicBlock, depth=depth, quant=quantize, noise=noise,
                              bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                              step_setup=step_setup, act_bitwidth=act_bitwidth, act_quant=act_quant,
                              normalize=normalize, noise_mask=noise_mask)


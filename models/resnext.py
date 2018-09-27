import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import quantize
import numpy as np
import torch.nn.functional as F
import collections
import actquant

__all__ = ['resnext']


def depBatchNorm2d(exists, *kargs, **kwargs):
    if exists:
        return nn.BatchNorm2d(*kargs, **kwargs)
    else:
        return lambda x: x


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_()  # The original initialization in class _BatchNorm
            m.bias.data.zero_()  # The original initialization in class _BatchNorm

            # m.weight.data.fill_(1)
            # m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            m.weight.data.normal_(0, math.sqrt(2. / n))


def partition_net(model, layers):
    for m in model.modules():
        if (isinstance(m, torch.nn.Conv2d) or isinstance(m,
                                                         torch.nn.Linear)):  # or isinstance(m, torch.nn.BatchNorm2d)):
            param = m
            if m.bias is not None:
                layers[m] = (param.weight.data, param.bias.data)
            else:
                layers[m] = param.weight.data

    return layers


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 batch_norm=True, act_bitwidth=32, act_quant=False, act_noise=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, planes, eps=1e-05)

        if act_bitwidth == 32:
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.relu1 = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
        self.conv2 = conv3x3(planes, planes, bias=not batch_norm)
        self.bn2 = depBatchNorm2d(batch_norm, planes, eps=1e-05)
        self.downsample = downsample
        if act_bitwidth == 32:
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.relu2 = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
        self.stride = stride

        # print(self.conv1)

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
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, batch_norm=True, act_bitwidth=32, act_quant=False,
                 act_noise=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, planes, eps=1e-05)

        if act_bitwidth == 32:
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.relu1 = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=not batch_norm, groups=32)
        self.bn2 = depBatchNorm2d(batch_norm, planes, eps=1e-05)

        if act_bitwidth == 32:
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.relu2 = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
        self.conv3 = nn.Conv2d(
            planes, planes * 2, kernel_size=1, bias=not batch_norm)
        self.bn3 = depBatchNorm2d(batch_norm, planes * 2, eps=1e-05)

        if act_bitwidth == 32:
            self.relu3 = nn.ReLU(inplace=True)
        else:
            self.relu3 = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
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


class PlainDownSample(nn.Module):

    def __init__(self, input_dims, output_dims, stride):
        super(PlainDownSample, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.stride = stride
        self.downsample = nn.AvgPool2d(stride)
        self.zero = Variable(torch.Tensor(1, 1, 1, 1).cuda(), requires_grad=False)

    def forward(self, inputs):
        ds = self.downsample(inputs)
        zeros_size = [ds.size(0), self.output_dims -
                      ds.size(1), ds.size(2), ds.size(3)]
        return torch.cat([ds, self.zero.expand(*zeros_size)], 1)


def check_nans(tensor, stri):
    my_np = tensor.data.cpu().numpy()
    if np.isnan(my_np).any():
        for i in range(tensor.size()[0]):
            if np.isnan(my_np[i]).any():
                for j in range(tensor.size()[1]):
                    if np.isnan(my_np[i][j]).any():
                        print(stri, i, j, tensor[i][j])
                        return True
    return False


# class ActQuant(nn.Module):

#     def __init__(self, quatize_during_training=False, noise_during_training=False, quant=False, noise=False, bitwidth=32):
#         super(ActQuant, self).__init__()
#         self.quant = quant
#         self.noise = noise
#         self.bitwidth = bitwidth
#         self.quatize_during_training = quatize_during_training
#         self.noise_during_training = noise_during_training

#     def update_stage(self, quatize_during_training=False, noise_during_training=False):
#         self.quatize_during_training = quatize_during_training
#         self.noise_during_training = noise_during_training

#     def forward(self, input):

#         if self.quant and (not self.training or (self.training and self.quatize_during_training)):
#             x = quantize.act_quantize(input, bitwidth=self.bitwidth)
#         elif self.noise and self.training and self.noise_during_training:
#             x = quantize.act_noise(input, bitwidth=self.bitwidth, training=self.training)
#         else:
#             x = F.relu(input)


#         return x

class ResNeXt(nn.Module):

    def __init__(self, shortcut='B', quant=False, noise=False, bitwidth=32, step=2, quant_edges=True, act_bitwidth=True,
                 act_noise=True, step_setup=[15, 9]):
        super(ResNeXt, self).__init__()
        self.shortcut = shortcut
        self.quant = quant
        self.noise = noise
        self.bitwidth = bitwidth
        self.training_stage = 0
        self.step = step
        self.act_bitwidth = act_bitwidth
        self.act_noise = act_noise

    def _make_layer(self, block, planes, blocks, stride=1,
                    batch_norm=True, act_bitwidth=32, act_quant=False):
        downsample = None
        if self.shortcut == 'C' or \
                self.shortcut == 'B' and \
                (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = [nn.Conv2d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=not batch_norm)]
            if batch_norm:
                downsample.append(nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
            downsample = nn.Sequential(*downsample)
        else:
            downsample = PlainDownSample(
                self.inplanes, planes * block.expansion, stride)

        layers = []

        layers.append(block(inplanes=self.inplanes, planes=planes,
                            stride=stride, downsample=downsample, batch_norm=batch_norm, act_bitwidth=act_bitwidth,
                            act_quant=act_quant, act_noise=self.act_noise))
        #         layers.append(block(self.inplanes, planes,
        #                     stride, downsample, batch_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, batch_norm=batch_norm, act_bitwidth=act_bitwidth, act_quant=act_quant,
                      act_noise=self.act_noise))
            # layers.append(block(self.inplanes, planes, batch_norm=batch_norm))

        return nn.Sequential(*layers)

    def switch_stage(self):
        # self.training_stage = 1
        # for layer in self.layers_half_one:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        if self.training_stage + 1 >= self.step:
            return
        print("Switching stage")
        self.training_stage += 1
        for step in self.layers_steps[:self.training_stage]:
            for layer in step:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    for param in layer.parameters():
                        param.requires_grad = False
                elif isinstance(layer, actquant.ActQuant):
                    layer.quatize_during_training = True
                    layer.noise_during_training = False

        if self.act_noise:
            for layer in self.layers_steps[self.training_stage]:  # Turn on noise only for current stage
                if isinstance(layer, actquant.ActQuant):
                    layer.noise_during_training = True

    def forward(self, x):
        temp_saved = {}

        if self.quant and not self.training:
            temp_saved = quantize.backup_weights(self.layers_list, {})
            quantize.quantize(self.layers_list, bitwidth=self.bitwidth)

        elif self.noise and self.training:
            # if self.training_stage==0:
            #     temp_saved = quantize.backup_weights(self.layers_half_one,{})
            #     quantize.add_noise(self.layers_half_one, bitwidth=self.bitwidth, training=self.training)     

            # else:                
            #     temp_saved = quantize.backup_weights(self.layers_half_one,{})
            #     quantize.quantize(self.layers_half_one, bitwidth=self.bitwidth)

            #     temp_saved = quantize.backup_weights(self.layers_half_two,temp_saved)
            #     quantize.add_noise(self.layers_half_two, bitwidth=self.bitwidth, training=self.training) 

            temp_saved = quantize.backup_weights(self.layers_steps[self.training_stage], {})
            quantize.add_noise(self.layers_steps[self.training_stage], bitwidth=self.bitwidth, training=self.training)

            for i in range(self.training_stage):
                temp_saved = quantize.backup_weights(self.layers_steps[i], temp_saved)
                quantize.quantize(self.layers_steps[i], bitwidth=self.bitwidth)
                # self.print_max_min_params()

                # print(temp_saved.keys()) 

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

        if self.quant and not self.training:
            quantize.restore_weights(self.layers_list, temp_saved)

        # elif self.noise and self.training:
        #     if self.training_stage==0:
        #         quantize.restore_weights(self.layers_half_one, temp_saved)
        #     else:
        #         quantize.restore_weights(self.layers_half_one+self.layers_half_two, temp_saved)

        elif self.noise and self.training:
            quantize.restore_weights(self.layers_steps[self.training_stage], temp_saved)  # Restore the noised layers
            for i in range(self.training_stage):
                quantize.restore_weights(self.layers_steps[i], temp_saved)  # Restore the quantized layers

        return x


class ResNeXt_imagenet(ResNeXt):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3], batch_norm=True, shortcut='B', quant=False, noise=False,
                 bitwidth=32, step=2, act_bitwidth=32, act_quant=False, quant_edges=True,
                 act_noise=True, step_setup=[15, 9]):
        super(ResNeXt_imagenet, self).__init__(shortcut=shortcut, quant=quant, noise=noise, bitwidth=bitwidth,
                                               step=step, act_bitwidth=act_bitwidth, quant_edges=quant_edges,
                                               act_noise=act_noise, step_setup=step_setup)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, 64, eps=1e-05)

        if act_bitwidth == 32:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth)
        self.quant_edges = quant_edges
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0],
                                       batch_norm=batch_norm, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       batch_norm=batch_norm, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       batch_norm=batch_norm, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
                                       batch_norm=batch_norm, act_bitwidth=act_bitwidth, act_quant=act_quant)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        self.layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc]
        # self.layers = [self.conv1,self.layer1,self.layer2,self.layer3,self.layer4,self.fc]
        init_model(self)

        print(step_setup)
        self.stages = list(range(step_setup[0], 1000, step_setup[1]))
        # self.stages = list(range(15, 1000, 9))
        self.regime = [

            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-3},
            {'epoch': 60, 'lr': 1e-4, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-5}

        ]
        modules_list = list(self.modules())
        self.layers_list = [x for x in modules_list if
                            isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear) or isinstance(x, actquant.ActQuant)]
        if not self.quant_edges:
            self.layers_list = self.layers_list[1:-1]

        # chunk = len(self.layers_list)//self.step
        # self.layers_half_one=self.layers_list[:chunk]
        # self.layers_half_two=self.layers_list[chunk:]
        self.layers_steps = np.array_split(self.layers_list, self.step)

        if self.act_noise:
            for layer in self.layers_steps[0]:  # Turn on noise for first stage
                if isinstance(layer, actquant.ActQuant):
                    layer.noise_during_training = True
        # print('Initing ResNeXt with {} activation and {}-bit weights.'.format('{}-bit custom {} noise'.format(activation.bitwidth, 'with' if self.act_noise else 'without') if isinstance(activation, actquant.ActQuant) else activation, bitwidth))


class ResNeXt_cifar10(ResNeXt):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18, batch_norm=True):
        super(ResNeXt_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, 16, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n,
                                       batch_norm=not batch_norm)
        self.layer2 = self._make_layer(block, 32, n, stride=2,
                                       batch_norm=not batch_norm)
        self.layer3 = self._make_layer(block, 64, n, stride=2,
                                       batch_norm=not batch_norm)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 5e-2,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 3},
            {'epoch': 5},

            {'epoch': 7, 'lr': 1e-2},

            {'epoch': 9},

            {'epoch': 11, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 13, 'lr': 1e-4}

            # {'epoch': 0, 'optimizer': 'SGD', 'lr': 5e-2,
            #  'weight_decay': 1e-4, 'momentum': 0.9},
            # {'epoch': 15},
            # {'epoch': 24},

            # {'epoch': 33, 'lr': 1e-2},

            # {'epoch': 42},

            # {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            # {'epoch': 90, 'lr': 1e-4}

        ]


def resnext(**kwargs):
    num_classes, depth, dataset, batch_norm, shortcut, quantize, noise, bitwidth, step, act_bitwidth, act_noise, act_quant, quant_edges, step_setup = map(
        kwargs.get,
        ['num_classes', 'depth', 'dataset', 'batch_norm', 'shortcut', 'quantize', 'noise', 'bitwidth', 'step',
         'act_bitwidth', 'act_noise', 'act_quant', 'quant_edges', 'step_setup'])
    dataset = dataset or 'cifar10'
    shortcut = shortcut or 'B'

    # quantize = getattr(kwargs, 'quantize', True)
    # noise = getattr(kwargs, 'noise', True)
    # bitwidth = getattr(kwargs, 'bitwidth', 5)

    if batch_norm is None:
        batch_norm = True

    # if act_bitwidth == 32:
    #     act = nn.ReLU(inplace=True)
    # else:
    #     act = actquant.ActQuant(quant=act_quant, noise=act_noise, bitwidth= act_bitwidth)

    # batch_norm = False    
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        # depth = depth or 18
        if depth == 18:
            return ResNeXt_imagenet(num_classes=num_classes,
                                    block=BasicBlock, layers=[2, 2, 2, 2],
                                    batch_norm=batch_norm, shortcut=shortcut, quant=quantize, noise=noise,
                                    bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                    quant_edges=quant_edges,
                                    act_noise=act_noise, step_setup=step_setup)
        if depth == 34:
            return ResNeXt_imagenet(num_classes=num_classes,
                                    block=BasicBlock, layers=[3, 4, 6, 3],
                                    batch_norm=batch_norm, shortcut=shortcut, quant=quantize, noise=noise,
                                    bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                    quant_edges=quant_edges,
                                    act_noise=act_noise, step_setup=step_setup)
        if depth == 50:
            return ResNeXt_imagenet(num_classes=num_classes,
                                    block=Bottleneck, layers=[3, 4, 6, 3],
                                    batch_norm=batch_norm, shortcut=shortcut, quant=quantize, noise=noise,
                                    bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                    quant_edges=quant_edges,
                                    act_noise=act_noise, step_setup=step_setup)
        if depth == 101:
            return ResNeXt_imagenet(num_classes=num_classes,
                                    block=Bottleneck, layers=[3, 4, 23, 3],
                                    batch_norm=batch_norm, shortcut=shortcut, quant=quantize, noise=noise,
                                    bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                    quant_edges=quant_edges,
                                    act_noise=act_noise, step_setup=step_setup)
        if depth == 152:
            return ResNeXt_imagenet(num_classes=num_classes,
                                    block=Bottleneck, layers=[3, 8, 36, 3],
                                    batch_norm=batch_norm, shortcut=shortcut, quant=quantize, noise=noise,
                                    bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                    quant_edges=quant_edges,
                                    act_noise=act_noise, step_setup=step_setup)

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 56
        return ResNeXt_cifar10(num_classes=num_classes,
                               block=BasicBlock, depth=depth, batch_norm=batch_norm)

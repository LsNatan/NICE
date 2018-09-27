import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable
import math
import torch.nn.functional as F
import quantize
import numpy as np
import actquant
from uniq import UNIQNet

__all__ = ['mobilenet']


def nearby_int(n):
    return int(round(n))

class Conv2d(nn.Conv2d):
    """ Create registered buffer for layer base for quantization"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.register_buffer('layer_b', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('layer_basis', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('initial_clamp_value', torch.ones(1))  # Attempt to enable multi-GPU


    def forward(self, input):

        output = F.conv2d(input, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        return output


class Linear(nn.Linear):
    """ Create registered buffer for layer base for quantization"""

    def __init__(self, in_features, out_features, bias=True):

        super(Linear, self).__init__(in_features, out_features, bias=True)
        self.register_buffer('layer_b', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('layer_basis', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('initial_clamp_value', torch.ones(1))  # Attempt to enable multi-GPU

    def forward(self, input):

        output = F.linear(input, self.weight, self.bias)

        return output


class DepthwiseSeparableFusedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act_bitwidth=32, act_quant=False,
                 act_noise=False, uniq=True):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        if act_bitwidth == 32:
            self.components = nn.Sequential(
                Conv2d(in_channels, in_channels, kernel_size,
                          stride=stride, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                # nn.ReLU(inplace=True),
                actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=uniq),

                Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
                actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=uniq)
            )
        else:
            self.components = nn.Sequential(
                Conv2d(in_channels, in_channels, kernel_size,
                          stride=stride, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=uniq),

                Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=uniq)
            )

    def forward(self, x):
        return self.components(x)


class MobileNet(UNIQNet):

    def __init__(self, quant_epoch_step,quant_start_stage, alpha=1.0, shallow=False, num_classes=1000, quant=False, noise=False, bitwidth=32, step=2,
                 quant_edges=True, act_bitwidth=32, act_noise=True, step_setup=[15, 9], act_quant=False, uniq=True,
                 normalize=False):
        super(MobileNet, self).__init__(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage, quant=quant, noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                     act_noise=act_noise, step_setup=step_setup, act_bitwidth=act_bitwidth,
                                     act_quant=act_quant, uniq=uniq)

        # if act_bitwidth == 32:
        #     act = nn.ReLU(inplace=True)
        # else:
        self.preprocess =  lambda x: x
        act = actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=self.uniq)

        layers = [
            Conv2d(3, nearby_int(alpha * 32),
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nearby_int(alpha * 32)),
            act,

            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 32), nearby_int(alpha * 64), kernel_size=3, padding=1, act_bitwidth=act_bitwidth,
                act_quant=act_quant, act_noise=act_noise, uniq=uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 64), nearby_int(alpha * 128), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 128), nearby_int(alpha * 128), kernel_size=3, padding=1,
                                          act_bitwidth=act_bitwidth, act_quant=act_quant, act_noise=act_noise, uniq=uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 128), nearby_int(alpha * 256), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise,uniq=uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 256), nearby_int(alpha * 256), kernel_size=3, padding=1,
                                          act_bitwidth=act_bitwidth, act_quant=act_quant, act_noise=act_noise, uniq=uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 256), nearby_int(alpha * 512), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise,uniq=uniq)
        ]
        if not shallow:
            # 5x 512->512 DW-separable convolutions
            layers += [
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=uniq),
            ]
        layers += [
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 1024), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=uniq),
            # Paper specifies stride-2, but unchanged size.
            # Assume its a typo and use stride-1 convolution
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 1024), nearby_int(alpha * 1024), kernel_size=3, stride=1,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=uniq)
        ]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = Linear(nearby_int(alpha * 1024), num_classes)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-3,
             # 'weight_decay': 5e-4, 'momentum': 0.9},
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 5e-3},
            {'epoch': 60, 'lr': 5e-4, 'weight_decay': 0},
            {'epoch': 90, 'lr': 5e-5}
        ]


    def forward(self, x):
        x = self.preprocess(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

"""
class MobileNet(nn.Module):

    def __init__(self, alpha=1.0, shallow=False, num_classes=1000, quant=False, noise=False, bitwidth=32, step=2,
                 quant_edges=True, act_bitwidth=32, act_noise=True, step_setup=[15, 9], act_quant=False, uniq=True):
        super(MobileNet, self).__init__()

        self.quant = quant
        self.noise = noise
        self.bitwidth = bitwidth
        self.training_stage = 0
        self.step = step
        self.act_bitwidth = act_bitwidth
        self.act_noise = act_noise
        self.quant_edges = quant_edges
        self.uniq = uniq

        if act_bitwidth == 32:
            act = nn.ReLU(inplace=True)
        else:
            act = actquant.ActQuantBuffers(quant=act_quant, noise=act_noise, bitwidth=act_bitwidth, uniq=self.uniq)

        layers = [
            nn.Conv2d(3, nearby_int(alpha * 32),
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nearby_int(alpha * 32)),
            act,

            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 32), nearby_int(alpha * 64), kernel_size=3, padding=1, act_bitwidth=act_bitwidth,
                act_quant=act_quant, act_noise=act_noise, uniq=self.uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 64), nearby_int(alpha * 128), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=self.uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 128), nearby_int(alpha * 128), kernel_size=3, padding=1,
                                          act_bitwidth=act_bitwidth, act_quant=act_quant, act_noise=act_noise, uniq=self.uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 128), nearby_int(alpha * 256), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise,uniq=self.uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 256), nearby_int(alpha * 256), kernel_size=3, padding=1,
                                          act_bitwidth=act_bitwidth, act_quant=act_quant, act_noise=act_noise, uniq=self.uniq),
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 256), nearby_int(alpha * 512), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise,uniq=self.uniq)
        ]
        if not shallow:
            # 5x 512->512 DW-separable convolutions
            layers += [
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=self.uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=self.uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=self.uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise, uniq=self.uniq),
                DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 512), kernel_size=3,
                                              padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                              act_noise=act_noise),
            ]
        layers += [
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 512), nearby_int(alpha * 1024), kernel_size=3, stride=2,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=self.uniq),
            # Paper specifies stride-2, but unchanged size.
            # Assume its a typo and use stride-1 convolution
            DepthwiseSeparableFusedConv2d(nearby_int(alpha * 1024), nearby_int(alpha * 1024), kernel_size=3, stride=1,
                                          padding=1, act_bitwidth=act_bitwidth, act_quant=act_quant,
                                          act_noise=act_noise, uniq=self.uniq)
        ]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(nearby_int(alpha * 1024), num_classes)

        print(step_setup)
        self.stages = list(range(step_setup[0], 1000, step_setup[1]))
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-3,
             # 'weight_decay': 5e-4, 'momentum': 0.9},
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 5e-3},
            {'epoch': 60, 'lr': 5e-4, 'weight_decay': 0},
            {'epoch': 90, 'lr': 5e-5}
        ]
        # self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    30: {'lr': 1e-2},
        #    60: {'lr': 1e-3},
        #    90: {'lr': 1e-4}
        # }

        modules_list = list(self.modules())
        self.layers_list = [x for x in modules_list if
                            isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear) or isinstance(x, actquant.ActQuantBuffers)]
        if not self.quant_edges:
            self.layers_list = self.layers_list[1:-1]

        self.layers_steps = np.array_split(self.layers_list, self.step)

        if self.act_noise:
            for layer in self.layers_steps[0]:  # Turn on noise for first stage
                if isinstance(layer, actquant.ActQuantBuffers):
                    layer.noise_during_training = True

    def switch_stage(self):
        if self.training_stage + 1 >= self.step:
            return
        print("Switching stage")
        self.training_stage += 1
        for step in self.layers_steps[:self.training_stage]:
            for layer in step:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    for param in layer.parameters():
                        param.requires_grad = False
                elif isinstance(layer, actquant.ActQuantBuffers):
                    layer.quatize_during_training = True
                    layer.noise_during_training = False

        if self.act_noise:
            for layer in self.layers_steps[self.training_stage]:  # Turn on noise only for current stage
                if isinstance(layer, actquant.ActQuantBuffers):
                    layer.noise_during_training = True

    def forward(self, x):

        temp_saved = {}

        if self.quant and not self.training:
            temp_saved = quantize.backup_weights(self.layers_list, {})
            quantize.quantize(self.layers_list, bitwidth=self.bitwidth)

        elif self.noise and self.training:

            temp_saved = quantize.backup_weights(self.layers_steps[self.training_stage], {})
            quantize.add_noise(self.layers_steps[self.training_stage], bitwidth=self.bitwidth, training=self.training)

            for i in range(self.training_stage):
                temp_saved = quantize.backup_weights(self.layers_steps[i], temp_saved)
                quantize.quantize(self.layers_steps[i], bitwidth=self.bitwidth)

        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.quant and not self.training:
            quantize.restore_weights(self.layers_list, temp_saved)

        elif self.noise and self.training:
            quantize.restore_weights(self.layers_steps[self.training_stage], temp_saved)  # Restore the noised layers
            for i in range(self.training_stage):
                quantize.restore_weights(self.layers_steps[i], temp_saved)  # Restore the quantized layers
        return x
"""
def mobilenet(**kwargs):
    num_classes, depth, dataset, batch_norm, shortcut, quantize, noise, bitwidth, step, \
    act_bitwidth, act_noise, act_quant, quant_edges, step_setup, uniq , quant_epoch_step,quant_start_stage, \
    normalize= map(kwargs.get,
                                                                      ['num_classes', 'depth', 'dataset', 'batch_norm',
                                                                       'shortcut', 'quantize', 'noise', 'bitwidth',
                                                                       'step', 'act_bitwidth', 'act_noise', 'act_quant',
                                                                       'quant_edges', 'step_setup', 'uniq',
                                                                       'quant_epoch_step','quant_start_stage',
                                                                       'normalize'])

    return MobileNet(quant_epoch_step=quant_epoch_step, quant_start_stage=quant_start_stage, quant=quantize, noise=noise, bitwidth=bitwidth, step=step, act_bitwidth=act_bitwidth,
                     act_quant=act_quant, quant_edges=quant_edges, act_noise=act_noise, step_setup=step_setup,
                     uniq=uniq, normalize=normalize)

# if __name__ == '__main__':
#    from time import time
#    model = model().cuda()
#    x = Variable(torch.rand(16, 3, 224, 224).cuda())
#    t = time()
#   y = model(x)
#   print(time() - t)

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
import numpy as np

import actquant
import quantize
from layers import ResConvLayer , Conv2d
from uniq import UNIQNet
#from quantize import quantize


class DenoisingNet(UNIQNet):
    """DenoisingNet implementation.
    """

    def __init__(self, in_channels=3, num_denoise_layers=20, num_filters=64, activation=nn.ReLU,  quant=True, noise=True,
                 bitwidth=32, use_cuda=True, quant_edges=True, act_quant = False, act_noise=False, act_bitwidth=8 , quant_epoch_step=80,
                 quant_start_stage=0, weight_relu=False,weight_grad_after_quant=False, random_inject_noise = False, std_weight_clamp=2.3,std_act_clamp=6.9,step=19, wrpn=False,quant_first_layer=True,num_of_layers_each_step=2):
        """DenoisingNet constructor.

        Arguments:
            in_channels (int, optional): number of channels in the input tensor. Default is 3 for RGB image inputs.
            num_denoise_layers (int, optional): number of denoising layers
            num_filters (int, optional): number of filters in denoising layers

        """

        super(DenoisingNet, self).__init__(quant_epoch_step=quant_epoch_step,quant_start_stage=quant_start_stage, quant=quant, noise=noise, bitwidth=bitwidth, step=step, quant_edges=quant_edges,
                                     act_noise=act_noise, step_setup=None, act_bitwidth=act_bitwidth,
                                     act_quant=act_quant, uniq=False,std_weight_clamp=std_weight_clamp,wrpn=wrpn,std_act_clamp=std_act_clamp,quant_first_layer = quant_first_layer,num_of_layers_each_step=num_of_layers_each_step)

        self.num_filters = num_filters

        self.activation = activation

        self.num_denoise_layers = num_denoise_layers
        self.in_channels = in_channels

        self.input_padder = nn.ReflectionPad2d(1)
        self.conv1 = Conv2d(in_channels, num_filters - 3, kernel_size=3)

        self.nonlinearity1 = self.get_feature_activation()



        self.denoising = self._make_denoising()
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def get_feature_activation(self):
        if self.wrpn:
            return actquant.ActQuantWRPN(act_quant=self.act_quant,  act_bitwidth=self.act_bitwidth )
        else:
            return actquant.ActQuantBuffers(quant=self.act_quant, bitwidth=self.act_bitwidth)


    def _make_denoising(self):
        modules = OrderedDict()
        stage_name = "Denoising"

        # add denoising layers
        for i in range(self.num_denoise_layers - 2):
            layer_num = i + 1
            name = stage_name + "_{}".format(layer_num)

            module = ResConvLayer(self.num_filters, self.num_filters, self.get_feature_activation(), 'reflect' , None, act_quant = self.act_quant, act_bitwidth=self.act_bitwidth)
            modules[name] = module
        # Last module has different number of outputs
        layer_num = self.num_denoise_layers - 1
        denoise_layer = ResConvLayer(self.num_filters, self.in_channels, self.get_feature_activation(), 'reflect', None, act_quant = self.act_quant, act_bitwidth=self.act_bitwidth)
        modules[stage_name + str(layer_num)] = denoise_layer

        return nn.Sequential(modules)

    def forward(self, image):

        x = self.nonlinearity1(self.conv1(self.input_padder(image)))
        image_out = self.denoising(torch.cat((x, image), 1))

        # Quantizing input image
        image_quant_scale = (2**16-1)/(torch.min(image) - torch.max(image))
        quant_image = torch.round(image_quant_scale * image)/image_quant_scale

        final_image = image_out + quant_image
        # final_image = image_out + image

        return final_image



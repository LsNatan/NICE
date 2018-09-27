import torch
import torch.nn as nn
import numpy as np
import quantize
import actquant

class Conv2d(nn.Conv2d):
    """ Create registered buffer for layer base for quantization"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.register_buffer('layer_b', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('layer_basis', torch.ones(1))  # Attempt to enable multi-GPU
        self.register_buffer('initial_clamp_value', torch.ones(1))  # Attempt to enable multi-GPU

class ResConvLayer(nn.Module):
    def __init__(self, input_filters, out_filters=32, nonlinearity=nn.ReLU, padding='reflect', image_nonlinearity=nn.Tanh, act_quant=False, act_bitwidth=8):
        super(ResConvLayer, self).__init__()

        self.out_filters = out_filters
        self.padding = padding
        self.act_quant = act_quant
        self.act_bitwidth = act_bitwidth

        if self.padding == 'reflect':
            self.padder = nn.ReflectionPad2d(1)
        if self.out_filters > 3:
            self.conv_features = Conv2d(input_filters, out_filters - 3, kernel_size=3,
                                           padding=1 if padding == 'zeros' else 0)
            self.features_activation = nonlinearity
        self.conv_image = Conv2d(input_filters, 3, kernel_size=3, padding=1 if padding == 'zeros' else 0)
        self.image_activation = image_nonlinearity() if image_nonlinearity is not None else None
        self.pic_act_quant =  actquant.ActQuantDeepIspPic(act_quant=act_quant,act_bitwidth = 8) # the picture is always quant to 8 bit
        self.dropout = nn.Dropout2d(p=0.05)

    def forward(self, input):
        image_input = input[:, -3:, :, :]
        # input = torch.cat((features_input, image_input), 1)


        if self.padding == 'reflect':
            input_padded = self.padder(input) #TODO we can try doing reflect to image , and zero to feature

        if (self.act_quant): # TODO approve and use ActQuant. right now it's not layer by layer
            feature_padded_input = input_padded[:, :61, :, :]
            image_padded_input = input_padded[:, -3:, :, :]

            #quant picture toward conv
            #quant_image_input = quantize.act_clamp_pic(image_padded_input)
            #quant_image_input = quantize.ActQuantizeImprovedWrpnForPic.apply( quant_image_input)
            #quant_image_input = actquant.act_quantize_pic_deep_isp(image_padded_input , self.act_bitwidth)
            quant_image_input = self.pic_act_quant(image_padded_input)

            input_padded = torch.cat((feature_padded_input, quant_image_input), 1)

        image_out = self.conv_image(input_padded)
        #image_out = self.conv_image(input)

        if self.image_activation is not None:
            image_out = self.image_activation(image_out)

        image_out = image_out / 8 + image_input

        #image_out = torch.clamp(image_out , -0.5, 0.5) #TODO try to add it in training

        if self.out_filters > 3:


            features_out = self.features_activation(self.conv_features(input_padded))

            #features_out = self.features_activation(self.conv_features(input))



            features_out = self.dropout(features_out)

            return torch.cat((features_out, image_out), 1)
        else:
            return image_out

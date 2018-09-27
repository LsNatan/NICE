import math

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import actquant

sqrt_of_2 = math.sqrt(2)

eps = 1e-5


def my_mean(x):
    size = x.size()
    size_tensor = torch.Tensor([s for s in size])
    elements = torch.prod(size_tensor)
    return torch.sum(x) / elements


def my_std(x):
    size = x.size()
    size_tensor = torch.Tensor([s for s in size])
    elements = torch.FloatTensor([torch.prod(size_tensor)])
    x_min_mean_sq = (x - my_mean(x)) * (x - my_mean(x))
    std = torch.sqrt(torch.sum(x_min_mean_sq) / (elements - 1))
    return std[0]


def norm_cdf(x, mean, std):
    return 1. / 2 * (1. + torch.erf((x - mean) / (std * sqrt_of_2)))


def norm_icdf(x, mean, std):
    return mean + std * sqrt_of_2 * torch.erfinv(2. * x - 1)


def uni_cdf(x, mn, max):
    return (x - mn) / (max - mn)


def uni_icdf(x, min, max):
    return x * (max - min) + min


class quantize(object):

    def __init__(self, weight_bitwidth, act_bitwidth, weight_scale_factor,
                 std_weight_clamp=3, std_act_clamp=3, noise_mask=0.05):  # The default clamp is std

        # These are the hyperparamters of the quantizer
        self.noise_mask = noise_mask
        self.max_factor_of_weight_step = 8  # the factor can be 1-8 in each lavel , will be represented in 3 bits in HW
        self.max_factor_of_act_step = 8  # the factor can be 1-8 in each lavel , will be represented in 3 bits in HW
        self.std_act_clamp = std_act_clamp
        self.std_weight_clamp = std_weight_clamp
        self.layers_basis_dict = {}

        self.act_max_value = 0

        self.allow_activation_quant_with_factor = True  # when enable, need to divide in hardware
        self.improvment_to_bin = False  # True
        self.bias_quantization = True
        self.num_of_bits_in_after_conv_add = 16 # weight_bitwidth + act_bitwidth + 2 if self.improvment_to_bin else weight_bitwidth + act_bitwidth + 1  # the granularity in add in *2 sensitive
        self.basis_weight_scale_factor = weight_scale_factor
        self.weight_bitwidth = weight_bitwidth
        self.act_bitwidth = act_bitwidth
        self.bn_bitwidth = 16
        self.hardware_clamp = False
        self.weight_max_int = 2 ** (
                    weight_bitwidth - 1) - 1 if not self.improvment_to_bin else 2 ** weight_bitwidth - 1  # for example for 4 bit in can be 7, int is [-7,...,7]
        self.act_max_int = 2 ** act_bitwidth - 1  # for example for 7 bit in can be 255, int is [0,...,255]
        self.quant_error = {}


    def add_improved_uni_noise(self, modules):
        for m in modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                weight_quant_step = None
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        d = m._parameters[p].device
                        if p == 'weight':
                            weight_quant_step = self.quant_step(m)
                            min_value = -self.weight_max_int * weight_quant_step  # .to(d)
                            max_value = self.weight_max_int * weight_quant_step  # .to(d)
                            y_p = uni_cdf(m._parameters[p].data, min_value, max_value)
                            noise_step = 1. / (
                                    2 ** (self.weight_bitwidth + 1) - 2) if self.improvment_to_bin else 1. / (2 ** (
                                    self.weight_bitwidth + 1) - 4)  # if not high_noise else 1. / (2 ** (bitwidth))
                            noise = y_p.clone().uniform_(-noise_step, noise_step)
                            y_out_p = uni_icdf(torch.clamp(y_p + noise, 0, 1), min_value, max_value)

                            # noise mask
                            p_noise_mask = self.noise_mask
                            mask = torch.bernoulli(
                                m._parameters[p].data.new(m._parameters[p].data.size()).fill_(p_noise_mask))
                            unmask = 1 - mask
                            quant_value, _ = self.quant_weight_wrpn_improved(m._parameters[p].data, m)
                            result = mask * y_out_p + unmask * quant_value
                            m._parameters[p].data = result.to(d)

                        if p == 'bias' and self.bias_quantization:
                            d = m._parameters[p].device
                            bias_max_value = self.get_bias_max_value(weight_quant_step)
                            min_value = -bias_max_value  # .to(d)
                            max_value = bias_max_value  # .to(d)
                            y_p = uni_cdf(m._parameters[p].data, min_value, max_value)
                            num_of_bits_in_after_conv_add = self.num_of_bits_in_after_conv_add
                            noise_step = 1. / (2 ** (num_of_bits_in_after_conv_add + 1))
                            noise = y_p.clone().uniform_(-noise_step, noise_step)
                            y_out_p = uni_icdf(torch.clamp(y_p + noise, 0, 1), min_value, max_value)

                            # dropout
                            p_noise_mask = self.noise_mask
                            mask = torch.bernoulli(
                                m._parameters[p].data.new(m._parameters[p].data.size()).fill_(p_noise_mask))
                            unmask = 1 - mask
                            quant_value = self.quant_bias_wrpn_improved(m._parameters[p].data, weight_quant_step)
                            result = mask * y_out_p + unmask * quant_value
                            m._parameters[p].data = result.to(d)



    def calc_b(self, wanted_clamp, layer_basis):
        b = wanted_clamp / wanted_clamp.new_tensor(layer_basis * self.weight_max_int)
        b = np.clip(b, 1, self.max_factor_of_weight_step)
        b = np.ceil(b)
        return b

    def basic_clamp(self, x):
        mean_p = x.mean()
        std_p = x.std()
        std_clamp = (mean_p > 0).float() * (mean_p + self.std_weight_clamp * std_p) + \
                    (mean_p < 0).float() * (mean_p - self.std_weight_clamp * std_p)
        clamp_value = abs(std_clamp)
        return clamp_value

    def get_weight_clamp_value(self, m):
        clamp_value = self.weight_max_int * m.layer_b.item() * m.layer_basis.item()  # self.get_basis_of_weight_clamp()
        # clamp_value = self.weight_max_int * self.layers_basis_dict[m] * self.get_basis_of_weight_clamp()
        return clamp_value


    def quant_step(self, m):
        return self.get_weight_clamp_value(m) / self.weight_max_int

    def get_weight_max_value(self, m):
        return self.weight_max_int * self.quant_step(m)

    def get_act_max_value_from_pre_calc_stats(self, modules):
        max_act_val = 0
        self_max_val= 0

        for layer in modules:
            if isinstance(layer, actquant.ActQuantBuffers):
                clamp_value = (layer.running_mean + self.std_act_clamp * layer.running_std)
                self_max_val = max(self_max_val,clamp_value)

                if (float(layer.clamp_val.data) == 0):  ##when we load model, we don't want to init this parameter
                    if self.hardware_clamp:
                        max_act_val = max(max_act_val, clamp_value)
                        self.act_max_value = max_act_val.__float__()
                        scaled_max_val_for_hw, p = self.calc_max_act_scale()
                        layer.clamp_val.data = self.calc_layer_act_clamp(clamp_value, p)  # scaled_max_val_for_hw
                    else:
                        max_act_val = max(max_act_val, clamp_value)
                        self.act_max_value = max_act_val.__float__()
                        layer.clamp_val.data = clamp_value
                else:
                    self.act_max_value = self_max_val.item()

                print("activation clamp: wanted clamp: ", clamp_value.__float__(), "acutal clamp: ",
                      layer.clamp_val.data)


        return

    def calc_layer_act_clamp(self, wanted_clamp, p):
        my_p = np.ceil(np.log2(wanted_clamp / (self.act_max_int * self.max_factor_of_act_step)))
        b = wanted_clamp / wanted_clamp.new_tensor((2 ** my_p) * self.act_max_int)
        b = np.clip(b, 1, self.max_factor_of_weight_step)

        if self.allow_activation_quant_with_factor:
            b = np.ceil(b)
        else:
            b = 2 ** np.ceil(math.log2(b))

        # return (b * (2 ** my_p) * self.act_max_int).__float__()
        return wanted_clamp.new_tensor((b * (2 ** my_p) * self.act_max_int))

    def assign_act_clamp_during_val(self, layers_list, print_clamp_val=False):
        """clamp_value = b*(2**p) in HW"""
        max_act_val = 0

        for layer in layers_list:
            if isinstance(layer, actquant.ActQuant):
                clamp_value = layer.clamp_val.data
                clamp_value1 = layer.running_mean + self.std_act_clamp * layer.running_std

                my_p = np.ceil(np.log2(clamp_value.item() / (self.act_max_int * self.max_factor_of_act_step)))
                b = clamp_value / clamp_value.new_tensor(2 ** my_p * self.act_max_int)
                b = np.clip(b, 1, self.max_factor_of_weight_step)
                # to use only
                if self.allow_activation_quant_with_factor:
                    b = np.ceil(b)
                else:
                    b = 2 ** np.ceil(math.log2(b))

                if self.hardware_clamp:
                    layer.clamp_val.data = clamp_value.new_tensor((b * (2 ** my_p) * self.act_max_int))
                    if print_clamp_val:
                        print("activation {} wanted clamp: {} actual clamp: {} = {}*2^{}"
                              .format(layer.layer_num, clamp_value.item(), layer.clamp_val.data.item(), b.item(), my_p))
                else:
                    layer.clamp_val.data = clamp_value

                    if print_clamp_val:
                        print("activation clamp: wanted clamp: ", clamp_value.item(), "actual clamp: ",
                              layer.clamp_val.data)

                max_act_val = max(max_act_val, clamp_value)
                self.act_max_value = max_act_val.item()

        return

    def assign_weight_clamp_during_val(self, layers_list, print_clamp_val=False):
        """clamp_value = b*(2**p) in HW"""
        layer_num = 0
        for layer in layers_list:
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                clamp_value = layer.layer_b * layer.layer_basis * self.weight_max_int

                my_p = np.ceil(np.log2(clamp_value.item() / (self.weight_max_int * self.max_factor_of_weight_step)))
                b = clamp_value / clamp_value.new_tensor(2 ** my_p * self.weight_max_int)
                b = np.clip(b, 1, self.max_factor_of_weight_step)
                # to use only
                if self.allow_activation_quant_with_factor:
                    b = np.ceil(b)
                else:
                    b = 2 ** np.ceil(math.log2(b))

                if self.hardware_clamp:
                    layer.layer_b.data = clamp_value.new_tensor(b)
                    layer.layer_basis.data = clamp_value.new_tensor(2 ** my_p)
                    if print_clamp_val:
                        print("weight {} wanted clamp: {} actual clamp: {} = {}*2^{}"
                              .format(layer_num, clamp_value,
                                      (layer.layer_b * layer.layer_basis * self.weight_max_int).item(), b.item(), my_p))
                else:

                    if print_clamp_val:
                        print("weight clamp: wanted clamp: ", clamp_value.item(), "actual clamp: ",
                              clamp_value)

                layer_num += 1

        return

    def get_act_max_value(self):
        return self.act_max_value

    def get_act_step(self):
        return self.get_act_max_value() / self.act_max_int

    def get_act_scale(self):
        return 1 / self.get_act_step()

    def calc_max_act_scale(self):
        p = np.ceil(np.log2((self.act_max_value / (self.act_max_int * self.max_factor_of_act_step))))
        max_act = self.act_max_int * self.max_factor_of_act_step * (2 ** p)
        return max_act, p

    def improved_wrpn_bias_scale(self, weight_quant_step):
        act_scale = self.get_act_scale()
        weight_scale = 1 / weight_quant_step
        return weight_scale * act_scale

    def quantize_clamp(self, modules):
        for m in modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        d = m._parameters[p].device
                        if (p == 'weight'):
                            m._parameters[p].data, weight_max_value = self.clamp_weights(m._parameters[p].data, m)
                        elif (p == 'bias') and self.bias_quantization:
                            m._parameters[p].data = self.bias_clamp(m._parameters[p].data,
                                                                    weight_max_value / self.weight_max_int)

    def clamp_weights(self, x, m):
        weight_max_value = self.get_weight_max_value(m)  # .item()
        return torch.clamp(x, -weight_max_value, weight_max_value), weight_max_value

    def quant_weight_wrpn_improved(self, x, m):
        x, weight_max_value = self.clamp_weights(x, m)
        weight_scale = self.weight_max_int / weight_max_value
        quantized_x = (1 / weight_scale) * self.round_to_int(x * weight_scale)
        #
        # if m not in self.quant_error:
        #     self.quant_error[m] = []
        # else:
        #     self.quant_error[m].append((quantized_x - x).view(-1).cpu().detach().numpy())

        return quantized_x.to(x.device), weight_max_value / self.weight_max_int

    def round_to_int(self, x):
        if self.improvment_to_bin:
            return 2 * torch.floor(x / 2) + 1
        else:
            return torch.round(x)

    def get_bias_max_value(self, quant_weight):
        num_of_bits_in_after_conv_add = self.num_of_bits_in_after_conv_add
        return self.get_act_step() * quant_weight * (2 ** (num_of_bits_in_after_conv_add - 1) - 1)

    def set_weight_basis(self, modules, layers_b_dict):
        # load from prev
        if layers_b_dict is not None and self.basis_weight_scale_factor is None:
            self.basis_weight_scale_factor = layers_b_dict['basis_weight_scale_factor']
            layer_index = 0
            for m in modules:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    dict_key = 'layer_' + str(layer_index)
                    m.layer_b, m.layer_basis = layers_b_dict[dict_key]
                    # self.layers_basis_dict[m] = layers_b_dict[dict_key]
                    layer_index += 1

            print("load statistic from loaded model - max weight in all layers after clamp is: ")

        # calc for first time
        if self.basis_weight_scale_factor is None:
            layers_b_dict = {}
            max_clamp = 0
            layer_num = 0
            for m in modules:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    for p in m._parameters:
                        if m._parameters[p] is not None:
                            if p == 'weight':
                                clamp_value = self.basic_clamp(m._parameters[p].data)
                                max_clamp = max(max_clamp, clamp_value)
                                if hasattr(m, 'initial_clamp_value') == False:
                                    raise AssertionError(
                                        'seems like buffer for initial_clamp_value, check model has buffer')

                                m.initial_clamp_value = clamp_value
                                layer_num += 1

            self.basis_weight_scale_factor = self.calc_basis(max_clamp)
            layers_b_dict['basis_weight_scale_factor'] = self.basis_weight_scale_factor

            print("calced weight statistic: ")
            layer_index = 0
            for m in modules:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    if hasattr(m, 'layer_b') == False:
                        raise AssertionError('seems like buffer for layer_b, check model has buffer')
                    if hasattr(m, 'layer_basis') == False:
                        raise AssertionError('seems like buffer for layer_basis, check model has buffer')
                    if self.hardware_clamp:
                        layer_basis = self.calc_basis(m.initial_clamp_value)
                        layer_b = self.calc_b(m.initial_clamp_value, layer_basis)

                        m.layer_b = layer_b
                        m.layer_basis = layer_basis
                        # m.register_buffer('layer_b', layer_b) # Attempt to enable multi-GPU
                        dict_key = 'layer_' + str(layer_index)
                        layers_b_dict[dict_key] = layer_b, layer_basis

                    else:
                        ## we set the wanted as the basis, and the b parameter to 1
                        m.layer_basis.data = m.initial_clamp_value / self.weight_max_int

                    print("layer : ", layer_index, "weight wanted clmap: ", m.initial_clamp_value.__float__(),
                          "actual: ",
                          (m.layer_basis * m.layer_b * self.weight_max_int).__float__(), "layer b:",
                          m.layer_b.__float__(),
                          "layer_basis: ", m.layer_basis.__float__())
                    layer_index += 1

        return layers_b_dict

    def calc_basis(self, wanted_clamp_value):
        return 2 ** np.ceil(np.log2(wanted_clamp_value / (self.weight_max_int * self.max_factor_of_weight_step)))

    def bias_clamp(self, x, weight_quant_step):
        bias_max_value = self.get_bias_max_value(weight_quant_step)
        x = torch.clamp(x, -bias_max_value, bias_max_value)
        return x

    def quant_bias_wrpn_improved(self, x, weight_quant_step):
        x = self.bias_clamp(x, weight_quant_step)
        bias_scale = self.improved_wrpn_bias_scale(weight_quant_step)
        return ((1 / bias_scale) * torch.round(x * bias_scale)).to(x.device)

    def quantize_uniform_improved(self, modules):
        for m in modules:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                weight_step = None
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        d = m._parameters[p].device
                        if (p == 'weight'):
                            m._parameters[p].data, weight_step = self.quant_weight_wrpn_improved(m._parameters[p].data,
                                                                                                 m)
                        elif (p == 'bias' and self.bias_quantization):
                            m._parameters[p].data = self.quant_bias_wrpn_improved(m._parameters[p].data, weight_step)

            elif isinstance(m, torch.nn.BatchNorm2d):
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        max_param = m._parameters[p].max()
                        min_param = m._parameters[p].min()

                        param_scale = (2 ** self.bn_bitwidth - 1) / (max_param-min_param)
                        m._parameters[p].data = torch.round(m._parameters[p] * param_scale) * 1 / param_scale


                for b in m._buffers:
                    if ('running_mean' in b or 'running_var' in b) and m._buffers[b] is not None:
                        max_param = m._buffers[b].max()
                        min_param = m._buffers[b].min()

                        buffer_scale = (2 ** self.bn_bitwidth - 1) / (max_param - min_param)
                        m._buffers[b].data = torch.round(m._buffers[b] * buffer_scale) * 1 / buffer_scale





    ###WRPN#################### use to compare deep isp
    def wrpn_weight_scale(self):
        return (2 ** (self.weight_bitwidth - 1) - 1)

    def quant_weight_wrpn(self, x, ):
        clamp = torch.clamp(x, -1, 1)
        return (1 / self.wrpn_weight_scale()) * torch.round(clamp * self.wrpn_weight_scale())

    def quantize_uniform_wrpn(self, moduls):
        for m in moduls:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        m._parameters[p].data = self.quant_weight_wrpn(m._parameters[p].data)



def backup_weights(modules, bk):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM) or \
                isinstance(m, actquant.ActQuant) or isinstance(m, torch.nn.BatchNorm2d):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    d = str(m._parameters[p].data.device)
                    if d not in bk:
                        bk[d] = {}
                    bk[d][(m, p)] = m._parameters[p].data.clone()
    return bk


def restore_weights(modules, bk):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM) or \
                isinstance(m, actquant.ActQuant) or isinstance(m, torch.nn.BatchNorm2d):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    m._parameters[p].data = bk[str(m._parameters[p].data.device)][(m, p)].clone()

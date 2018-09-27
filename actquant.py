import itertools
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import quantize


class ActQuant(nn.Module):
    layer_id = itertools.count()

    def __init__(self, quatize_during_training=False, noise_during_training=False, quant=False,
                 bitwidth=32):
        super(ActQuant, self).__init__()

        self.quant = quant
        assert (isinstance(bitwidth, int))
        self.bitwidth = bitwidth
        self.act_full_scale = 2 ** self.bitwidth - 1
        self.quatize_during_training = quatize_during_training
        self.noise_during_training = noise_during_training
        self.layer_num = next(self.layer_id)
        self.saved_stats = False
        self.gather_stats = False
        self.quant_error = []

        # For gathering statistics before training
        self.pre_training_statistics = False
        self.momentum = 0.9

        # For activation scaling
        self.max_factor_of_act_step = 8  # the factor can be 1-8 in each level , will be represented in 3 bits in HW


    def update_stage(self, quatize_during_training=False, noise_during_training=False):
        self.quatize_during_training = quatize_during_training
        self.noise_during_training = noise_during_training

    def plot_statistic(self, x):

        # plot histogram
        gaussian_numbers = x.view(-1).cpu().detach().numpy()
        plt.hist(gaussian_numbers, bins=256)
        file_name = 'activation_value_' + str(self.layer_num)
        if not os.path.isdir('./activation_stats'):
            os.mkdir('./activation_stats')

        file_name = os.path.join('./activation_stats', file_name + '.png')
        plt.savefig(file_name)
        plt.close()


    def act_clamp(self, x, clamp_val):
        x = F.relu(x) - F.relu(x - torch.abs(clamp_val))
        return x

    def forward(self, input):

        if self.quant and (not self.training or (self.training and self.quatize_during_training)):
            x = self.act_clamp(input, self.clamp_val)
            x = act_quant.apply(x, self.clamp_val, self.bitwidth)
            print('Activation layer {}  has clamp value {}'.format(self.layer_num, self.clamp_val.item()))

        else:
            # x = F.relu(input)
            # x = quantize.act_clamp(input)
            x = self.act_clamp(input, self.clamp_val)

            if not self.saved_stats and self.gather_stats:
                self.plot_statistic(x)
                self.saved_stats = True

        return x


def act_quant(x, act_max_value, bitwidth):
    act_scale = (2 ** bitwidth - 1) / act_max_value
    q_x = Round.apply(x * act_scale) * 1 / act_scale
    return q_x


class Round(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        round = (x).round()
        return round.to(x.device)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input, None, None


class ClampQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min, max, bin_num):
        ctx.save_for_backward(x, min, max, bin_num)
        # print(min, max)
        return torch.clamp(x, min.item(), max.item())

    @staticmethod
    def backward(ctx, grad_output):
        x, mn, mx, bin_num = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < mn] = 0
        grad_x[x > mx] = 0
        return grad_x, None, None, None
        grad_min = (grad_output[x <mn]).sum()-1 / (4 * bin_num)
        grad_max = (grad_output[x > mx]).sum()+1 / (4 * bin_num)
        #print(grad_min, grad_max)
        return grad_x, grad_min.expand_as(mn), grad_max.expand_as(mx), None


class ActQuantBuffers(ActQuant):  # This class exist to allow multi-gpu run

    def __init__(self, quatize_during_training=False, noise_during_training=False, quant=False,
                 bitwidth=32):
        super(ActQuantBuffers, self).__init__(quatize_during_training=quatize_during_training,
                                              noise_during_training=noise_during_training, quant=quant,
                                              bitwidth=bitwidth)

        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.zeros(1))
        self.clamp_val = Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):
        if self.pre_training_statistics:

            self.running_mean.to(input.device).detach().mul_(self.momentum).add_(
                input.mean() * (1 - self.momentum))

            self.running_std.to(input.device).detach().mul_(self.momentum).add_(
                input.std() * (1 - self.momentum))

            x = F.relu(input)

        elif self.quant and (not self.training or (self.training and self.quatize_during_training)):

            c_x = self.act_clamp(input, self.clamp_val)
            x = act_quant(c_x, self.clamp_val, self.bitwidth)

        else:
            if self.quant:
                x = self.act_clamp(input, self.clamp_val)
            else:
                x = F.relu(input)

            if not self.saved_stats and self.gather_stats:
                self.plot_statistic(x)
                self.saved_stats = True

        if False:
            self.print_clamp()

        return x

    def print_clamp(self):
        print('Activation layer {}  has clamp value {}'.format(self.layer_num, self.clamp_val.item()))


class ActQuantDeepIspPic(nn.Module):
    def __init__(self, act_quant=False, act_bitwidth=8, act_clamp=0.5):
        super(ActQuantDeepIspPic, self).__init__()
        self.act_quant = act_quant
        self.act_bitwidth = act_bitwidth
        self.act_clamp = act_clamp
        self.act_scale = 2 ** self.act_bitwidth - 1
        self.quatize_during_training = False
        self.noise_during_training = False

    def forward(self, x):
        if self.act_quant and (not self.training or (self.training and self.quatize_during_training)):
            x = torch.clamp(x, -self.act_clamp, self.act_clamp)
            x = 1 / self.act_scale * Round.apply(x * self.act_scale)
            return x
        else:
            if self.act_quant:
                return torch.clamp(x, -self.act_clamp, self.act_clamp)
            else:
                return x


###WRPN to compare in deep isp
class ActQuantWRPN(nn.Module):
    def __init__(self, act_quant=False, act_bitwidth=8, act_clamp=1):
        super(ActQuantWRPN, self).__init__()
        self.act_quant = act_quant
        self.act_bitwidth = act_bitwidth
        self.act_clamp = act_clamp
        self.act_scale = 2 ** self.act_bitwidth - 1
        self.quatize_during_training = False
        self.noise_during_training = False

    def forward(self, x):
        if self.act_quant and (not self.training or (self.training and self.quatize_during_training)):
            x = torch.clamp(x, 0, self.act_clamp)
            x = 1 / self.act_scale * Round.apply(x * self.act_scale)
            return x
        else:
            return F.relu(x)

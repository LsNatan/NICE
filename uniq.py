import numpy as np
import torch.nn as nn

import actquant
import quantize


def save_state(self, _):

    self.full_parameters = {}
    layers_list = self.layers_list()
    layers_steps = self.layers_steps()

    self.full_parameters = quantize.backup_weights(layers_list, {})

    if self.quant and not self.training and not self.statistics_phase:
        for i in range(len(layers_steps)):
            self.quantize.quantize_uniform_improved(layers_steps[i])

        if self.quantize.hardware_clamp:
            self.quantize.assign_act_clamp_during_val(layers_list)
            self.quantize.assign_weight_clamp_during_val(layers_list)

    elif self.quant and self.training:


        if self.allow_grad:
            for i in range(self.quant_stage_for_grads):
                self.quantize.quantize_uniform_improved(layers_steps[i])

        else:
            if self.noise:
                self.quantize.add_improved_uni_noise(layers_steps[self.training_stage])
            for i in range(self.training_stage):
                self.quantize.quantize_uniform_improved(layers_steps[i])


def restore_state(self, _, __):
    layers_list = self.layers_list()
    quantize.restore_weights(layers_list, self.full_parameters)

class UNIQNet(nn.Module):

    def __init__(self, quant_epoch_step,quant_start_stage, quant=False, noise=False, bitwidth=32, step=2,
                 quant_edges=True, act_noise=True, step_setup=[15, 9], act_bitwidth=32, act_quant=False, uniq=False,
                 std_act_clamp=5, std_weight_clamp=3.45, wrpn=False,quant_first_layer=False,
                 num_of_layers_each_step=1, noise_mask=0.05):
        super(UNIQNet, self).__init__()
        self.quant_epoch_step  = quant_epoch_step
        self.quant_start_stage = quant_start_stage
        self.quant = quant
        self.noise = noise
        self.wrpn = wrpn
        if isinstance(bitwidth, list):
            assert (len(bitwidth) == step)
            self.bitwidth = bitwidth
        else:
            self.bitwidth = [bitwidth for _ in range(step)]
        self.training_stage = 0
        self.step = step
        self.num_of_layers_each_step = num_of_layers_each_step
        self.act_noise = act_noise
        self.act_quant = act_quant
        self.act_bitwidth = act_bitwidth
        self.quant_edges = quant_edges
        self.quant_first_layer = quant_first_layer
        self.register_forward_pre_hook(save_state)
        self.register_forward_hook(restore_state)
        self.layers_b_dict = None
        self.noise_mask_init = 0. if not noise else  noise_mask
        self.quantize = quantize.quantize(bitwidth, self.act_bitwidth, None, std_act_clamp=std_act_clamp,
                                          std_weight_clamp=std_weight_clamp, noise_mask=self.noise_mask_init)
        self.statistics_phase = False
        self.allow_grad = False
        self.random_noise_injection = False

        self.open_grad_after_each_stage = True
        self.quant_stage_for_grads = quant_start_stage

        self.noise_level = 0
        self.noise_batch_counter = 0

    def layers_list(self):
        modules_list = list(self.modules())
        quant_layers_list =  [x for x in modules_list if
                isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear) or isinstance(x, actquant.ActQuant)
                              or isinstance(x, actquant.ActQuantDeepIspPic) or isinstance(x, actquant.ActQuantWRPN)
                              or isinstance(x, nn.BatchNorm2d)]

        if not self.quant_edges:
            if self.act_quant:
                quant_layers_list[-2].quant = False
                quant_layers_list = quant_layers_list[1:-2]

            else:
                quant_layers_list = quant_layers_list[1:-1]
        else:
            if not self.quant_first_layer:
                quant_layers_list = quant_layers_list[1:] #remove first weight. this mode quant last layer, but not first

        return quant_layers_list

    def layers_steps(self):
        split_layers =  self.split_one_layer_with_parameter_in_step()
        return split_layers


    def count_of_parameters_layer_in_list(self,list):
        counter = 0
        for layer in list:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                counter += 1
        return counter

    def split_one_layer_with_parameter_in_step(self):
        layers = self.layers_list()
        splited_layers = []
        split_step = []
        for layer in layers:

            if  (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) and self.count_of_parameters_layer_in_list(split_step) == self.num_of_layers_each_step:
                splited_layers.append(split_step)
                split_step = []
                split_step.append(layer)
            else:
                split_step.append(layer)

        #add left layers
        if len(split_step) > 0:
            splited_layers.append(split_step)

        return splited_layers

    def switch_stage(self, epoch_progress):
        """
        Switches the stage of network to the next one.
        :return:
        """

        layers_steps = self.layers_steps()
        max_stage = len( layers_steps )
        if self.training_stage >= max_stage + 1:
            return

        if self.open_grad_after_each_stage == False:
            if (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage > self.training_stage and self.training_stage < max_stage - 1):
                self.training_stage += 1
                print("Switching stage, new stage is: ", self.training_stage)
                for step in layers_steps[:self.training_stage]:
                    for layer in step:
                        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)\
                                or isinstance(layer, nn.BatchNorm2d):

                            for param in layer.parameters():
                                param.requires_grad = False
                        elif isinstance(layer, actquant.ActQuant) or isinstance(layer, actquant.ActQuantDeepIspPic) or isinstance(layer, actquant.ActQuantWRPN):
                            layer.quatize_during_training = True
                            layer.noise_during_training = False

                if self.act_noise:
                    for layer in layers_steps[self.training_stage]:  # Turn on noise only for current stage
                        if isinstance(layer, actquant.ActQuant) or isinstance(layer, actquant.ActQuantDeepIspPic) or isinstance(layer, actquant.ActQuantWRPN):
                            layer.noise_during_training = True
                return True

            elif (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage >  max_stage - 1 and self.allow_grad == False):
                self.allow_grad = True
                self.quant_stage_for_grads = self.training_stage + 1
                self.random_noise_injection = False
                print("Switching stage, allowing all grad to propagate. new stage is: ", self.training_stage)
                for step in layers_steps[:self.training_stage]:
                    for layer in step:
                        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                            for param in layer.parameters():
                                param.requires_grad = True
                return True
            return False

        else:

            if (np.floor( epoch_progress / self.quant_epoch_step) + self.quant_start_stage > self.training_stage and
                    self.training_stage < max_stage - 1):

                self.training_stage += 1
                print("Switching stage, new stage is: ", self.training_stage)
                for step in layers_steps[:self.training_stage]:
                    for layer in step:
                        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)\
                                or isinstance(layer, nn.BatchNorm2d):
                            for param in layer.parameters():
                                param.requires_grad = True
                        elif isinstance(layer, actquant.ActQuant) or isinstance(layer, actquant.ActQuantDeepIspPic) or isinstance(layer, actquant.ActQuantWRPN):
                            layer.quatize_during_training = True
                            layer.noise_during_training = False

                if self.act_noise:
                    for layer in layers_steps[self.training_stage]:  # Turn on noise only for current stage
                        if isinstance(layer, actquant.ActQuant) or isinstance(layer, actquant.ActQuantDeepIspPic) or isinstance(layer, actquant.ActQuantWRPN):
                            layer.noise_during_training = True

                self.allow_grad = False
                return True

            if (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage > max_stage - 1 and self.allow_grad == False):
                self.allow_grad = True
                self.quant_stage_for_grads = self.training_stage + 1
                self.random_noise_injection = False
                print("Switching stage, allowing all grad to propagate. new stage is: ", self.training_stage)

            return False


import argparse
import os
import torch
from torch.autograd import Variable


import numpy as np
from torch import nn
import quantize

def calc_loss(output , target , criterion , model, args):
    loss_for_psnr = criterion(output, target)
    loss = loss_for_psnr.clone()

    weight_decay_loss = Variable( torch.FloatTensor([0]))

    if (args.gpus is not None):
        weight_decay_loss = weight_decay_loss.cuda()

    #act_decay_loss = Variable( torch.FloatTensor([0]))

    is_weight_layer = False
    if (args.enable_decay):
        L1 =  nn.MSELoss(size_average=False) # nn.L1Loss(size_average=False) #

        if (args.gpus is not None):
            L1 = L1.cuda()

        #moduls = model.layers_steps[model.training_stage:]
        moduls = model.layers_steps[model.training_stage]

        for m in moduls:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
                is_weight_layer = True
                for p in m._parameters:
                    if m._parameters[p] is not None:
                        #quantize_target = quantize.quantize_for_decay(m._parameters[p], bitwidth=model.bitwidth)
                        #quantize_target = Variable( quantize_target , requires_grad = False )
                        #print(torch.max( torch.abs(m._parameters[p].data) ) )
                        decay_mask = (torch.abs(m._parameters[p].data) > 0.5 ).type(torch.FloatTensor)
                        if (args.gpus is not None):
                            decay_mask = decay_mask.cuda()

                        clamp_weight = torch.clamp( m._parameters[p].data , -0.1 , 0.1)
                        quantize_target =  Variable( clamp_weight * decay_mask , requires_grad = False )
                        mask_weights = m._parameters[p] * Variable(decay_mask)

                        if (args.gpus is not None):
                            quantize_target = quantize_target.cuda()

                        weight_decay_loss += L1(mask_weights , quantize_target)


    if (is_weight_layer):

        factor_weight = 1 if (weight_decay_loss.data[0] == 0) else Variable (loss_for_psnr.data * args.quant_decay / weight_decay_loss.data, requires_grad = False)

        #factor_act    = 1 if (act_decay_loss.data[0] == 0) else Variable (loss_for_psnr.data * 0.0 / act_decay_loss.data, requires_grad = False)

        loss_for_dcay = factor_weight * weight_decay_loss #+ factor_act * act_decay_loss
        loss += loss_for_dcay

        #print('loss_for_psnr: ', loss_for_psnr.data[0] , 'loss for decay: ',  loss_for_dcay.data[0], 'weight decay before factor: ', weight_decay_loss.data[0] , 'act: ', act_decay_loss.data[0])

    return loss_for_psnr , loss , weight_decay_loss
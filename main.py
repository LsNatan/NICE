import argparse
import csv
import logging
import os
import sys
from ast import literal_eval
from datetime import datetime
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm, trange
from collections import OrderedDict
import actquant

import models
from clr import CyclicLR  # Until it will be included in official PyTorch release
from data import get_dataset
from logger import CsvLogger
from preprocess import get_transform
from utils.log import save_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--datapath', metavar='DATA_PATH', default='./results', help='datasets dir')
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None, help='image input size')
parser.add_argument('--model_config', default='', help='additional architecture configuration')
parser.add_argument('--type', default='float32', help='type of tensor - e.g float16')
parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT', help='optimizer function used')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE', help='evaluate model FILE on validation set')
parser.add_argument('--no-quantization', action='store_true', default=False, help='disables quantization')
parser.add_argument('--no-noise', action='store_true', default=False, help='noise')
parser.add_argument('--bitwidth', default=32, type=int, metavar='N', help='Quantization bitwidth (default: 5)')

parser.add_argument('--scale', default=1, type=float, metavar='N', help='scale of MobileNet2')
parser.add_argument('--step', default=2, type=int, metavar='N',
                    help='portion of net to be quantized at second stage(default: 2)')
parser.add_argument('--depth', default=18, type=int, metavar='N', help='depth of the model(default: 18)')
parser.add_argument('--act-bitwidth', default=32, type=int, metavar='N',
                    help='Quantization activation bitwidth (default: 5)')
parser.add_argument('--no-act-quantization', action='store_true', default=False, help='disables quantization')
parser.add_argument('--start-from-zero', action='store_true', default=False, help='Start from epoch 0')
parser.add_argument('--no-quant-edges', action='store_true', default=False,
                    help='no quantization for first and last layers')
#parser.add_argument('--step-setup', default='15,9', help='start of steps and interval- e.g 0,1')
parser.add_argument('--quant_start_stage', default=0, type=int, metavar='N', help='from which level of quant to start')
parser.add_argument('--quant_epoch_step', type=float, default=1.0, help='hot often to change state of quant')


# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=1, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=20,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Optimization options
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122, 164],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--val_batch_size', default=64, type=int , help='val mini-batch size (default: 64)')


# NICE
parser.add_argument('--param-std-cutoff', type=float, default=3, help='how many std to include before cutoff')
parser.add_argument('--quant-dataloader', action='store_true', default=False, help='Load quantized data loader')
parser.add_argument('-sb', '--act_stats_batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--no_pre_process_normalize', action='store_true', default=False, help='normalize in the preprocess')
parser.add_argument('--noise_mask', type=float, default=0.05, help='Probability to add noise')



clamp_stats_dict = {}
cos_loss_dict = {}

def load_model(model, checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():

        name = k if k[0:7] != 'module.' else k[7:]  # remove `module. if needed (happen when the model created with DataParallel
        new_state_dict[name] = v if v.dim() > 1 or 'num_batches_tracked' in name else v*v.new_ones(1)

    # load params
    model.load_state_dict(new_state_dict, strict=False) #strict false in case the loaded doesn't have alll variables like running mean

def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.noise = not args.no_noise
    args.quant = not args.no_quantization
    args.act_quant = not args.no_act_quantization
    args.quant_edges = not args.no_quant_edges


    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'
    dtype = torch.float32

    args.step_setup = None

    model = models.__dict__[args.model]
    model_config = {'scale': args.scale, 'input_size': args.input_size, 'dataset': args.dataset,
                    'bitwidth': args.bitwidth, 'quantize': args.quant, 'noise': args.noise, 'step': args.step,
                    'depth': args.depth, 'act_bitwidth': args.act_bitwidth, 'act_quant': args.act_quant,
                    'quant_edges': args.quant_edges, 'step_setup': args.step_setup,
                    'quant_epoch_step': args.quant_epoch_step, 'quant_start_stage': args.quant_start_stage,
                    'normalize':  args.no_pre_process_normalize,
                    'noise_mask': args.noise_mask}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    # create model
    model = model(**model_config)
    logging.info("creating model %s", args.model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters: ", params)
    logging.info("created model with configuration: %s", model_config)
    print(model)


    data = None
    checkpoint_epoch=0
    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate, map_location=device)
        load_model(model, checkpoint)
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])

        print("loaded checkpoint {0} (epoch {1})".format(args.evaluate, checkpoint['epoch']))

    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            if not args.start_from_zero:
                args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            checkpoint_epoch = checkpoint['epoch']

            load_model(model, checkpoint)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpus is not None:
        model = torch.nn.DataParallel(model, [args.gpus[0]])  # Statistics need to be calculated on single GPU to be consistant with data among multiplr GPUs

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset, input_size=args.input_size, augment=True,
                               integer_values=args.quant_dataloader, norm=not args.no_pre_process_normalize),
        'eval': get_transform(args.dataset, input_size=args.input_size, augment=False,
                              integer_values=args.quant_dataloader, norm=not args.no_pre_process_normalize)
    }
    transform = getattr(model.module, 'input_transform', default_transform)


    val_data = get_dataset(args.dataset, 'val', transform['eval'], datasets_path=args.datapath)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_data = get_dataset(args.dataset, 'train', transform['train'], datasets_path=args.datapath)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    statistics_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.act_stats_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    model, criterion = model.to(device, dtype), criterion.to(device, dtype)
    if args.clr:
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)
    csv_logger_training_stats = os.path.join(save_path, 'training_stats.csv')

    # pre-training activation and parameters statistics calculation ####
    if check_if_need_to_collect_statistics(model):
        for layer in model.modules():
            if isinstance(layer, actquant.ActQuantBuffers):
                layer.pre_training_statistics = True  # Turn on pre-training activation statistics calculation
        model.module.statistics_phase = True

        validate(statistics_train_loader, model, criterion, device, epoch=0, num_of_batches=80, stats_phase=True)  # Run validation on training set for statistics
        model.module.quantize.get_act_max_value_from_pre_calc_stats(list(model.modules()))
        _ = model.module.quantize.set_weight_basis(list(model.modules()), None)

        for layer in model.modules():
            if isinstance(layer, actquant.ActQuantBuffers):
                layer.pre_training_statistics = False  # Turn off pre-training activation statistics calculation
        model.module.statistics_phase = False



    else:  # Maximal activation values still need to be derived from loaded stats
        model.module.quantize.assign_act_clamp_during_val(list(model.modules()), print_clamp_val=True)
        model.module.quantize.assign_weight_clamp_during_val(list(model.modules()), print_clamp_val=True)
        # model.module.quantize.get_act_max_value_from_pre_calc_stats(list(model.modules()))

    if args.gpus is not None:  # Return to Multi-GPU after statistics calculations
        model = torch.nn.DataParallel(model.module, args.gpus)
        model, criterion = model.to(device, dtype), criterion.to(device, dtype)

    # pre-training activation statistics calculation ####

    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, device, epoch=0)
        print("val_prec1: ", val_prec1)
        return

    # fast forward to curr stage
    for i in range(args.quant_start_stage):
        model.module.switch_stage(0)

    for epoch in trange(args.start_epoch, args.epochs + 1):

        if not isinstance(scheduler, CyclicLR):
            scheduler.step()

        #     scheduler.optimizer = optimizer
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, device, epoch, optimizer, scheduler,
            training_stats_logger=csv_logger_training_stats)

        for layer in model.modules():
            if isinstance(layer, actquant.ActQuantBuffers):
                layer.print_clamp()

        # evaluate on validation set

        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, device, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'layers_b_dict': model.module.layers_b_dict #TODO this doesn't work for multi gpu - need to del
        }, is_best, path=save_path)
        # New type of logging
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - val_prec1, 'val_error5': 1 - val_prec5,
                          'val_loss': val_loss, 'train_error1': 1 - train_prec1,
                          'train_error5': 1 - train_prec5, 'train_loss': train_loss})
        csv_logger.plot_progress(title=args.model+str(args.depth))
        csv_logger.write_text('Epoch {}: Best accuracy is {:.2f}% top-1'.format(epoch + 1, best_prec1 * 100.))


def check_if_need_to_collect_statistics(model):
    for layer in model.modules():
    # for layer in model.module.layers_list():
        if isinstance(layer, actquant.ActQuantBuffers):
            if hasattr(layer, 'running_std') and float(layer.running_std) != 0:
                return False

    return True

def forward(data_loader, model, criterion, device, epoch=0, num_of_batches=None, training=True, optimizer=None,
            scheduler=None, training_stats_logger=None, stats_phase=False):

    correct1, correct5 = 0, 0
    print_correct_1 , print_correct_5 = 0, 0
    print_batch_counter = 0
    quant_stage_counter = 0
    quant_stage_correct_1 = 0
    t = time.time()
    for batch_idx, (inputs, target) in enumerate(tqdm(data_loader)):
        if num_of_batches:
            if batch_idx > num_of_batches:  # Debug
                break
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()

        inputs, target = inputs.to(device=device), target.to(device=device)

        if (training):
            epoch_progress = epoch + batch_idx/len(data_loader)
            stage_switch = model.module.switch_stage(epoch_progress)
            if stage_switch:
                quant_stage_counter = 0
                quant_stage_correct_1 = 0


        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        if training:
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        quant_stage_correct_1 += corr[0]
        print_correct_1 += corr[0]
        print_correct_5 += corr[1]
        print_batch_counter += output.shape[0]
        quant_stage_counter += output.shape[0]
        if batch_idx % args.print_freq == 0:
            if stats_phase:
                tqdm.write('Stats phase :  [{}/{} ({:.0f}%)]\tLoss: {:.6f}. Top-1 accuracy: {:.2f}%({:.2f}%). '
                           'Top-5 accuracy: '
                           '{:.2f}%({:.2f}%).'.format(batch_idx, len(data_loader),
                                                      100. * batch_idx / len(data_loader), loss.item(),
                                                      100. * print_correct_1 / print_batch_counter,
                                                      100. * correct1 / (args.act_stats_batch_size * (batch_idx + 1)),
                                                      100. * print_correct_5 / print_batch_counter,
                                                      100. * correct5 / (args.act_stats_batch_size * (batch_idx + 1))))

            elif training:
                tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. Top-1 accuracy: {:.2f}%({:.2f}%). '
                           'Top-5 accuracy: {:.2f}%({:.2f}%). \t'
                           'lr: {:.2e}.'.format(epoch, batch_idx, len(data_loader),
                                                100. * batch_idx / len(data_loader), loss.item(),
                                                100. * print_correct_1 / print_batch_counter,
                                                100. * correct1 / (args.batch_size * (batch_idx + 1)),
                                                100. * print_correct_5 / print_batch_counter,
                                                100. * correct5 / (args.batch_size * (batch_idx + 1)),
                                                scheduler.get_lr()[0] if scheduler is not None else 0))

                dur = time.time() - t
                with open(training_stats_logger, 'a') as f: #TODO add title
                    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{} \n'.format(epoch, batch_idx, len(data_loader),
                                                epoch * len(data_loader) + batch_idx,
                                                100. * batch_idx / len(data_loader), loss.item(),
                                                100. * print_correct_1 / print_batch_counter,
                                                100. * correct1 / (args.batch_size * (batch_idx + 1)),
                                                100. * print_correct_5 / print_batch_counter,
                                                100. * correct5 / (args.batch_size * (batch_idx + 1)),
                                                scheduler.get_lr()[0] if scheduler is not None else 0,
                                                dur ,
                                                100. * quant_stage_correct_1 / quant_stage_counter,

                                                                              )
                            )


            else:
                tqdm.write('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. Top-1 accuracy: {:.2f}%({:.2f}%). '
                           'Top-5 accuracy: '
                           '{:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(data_loader),
                                                      100. * batch_idx / len(data_loader), loss.item(),
                                                      100. * print_correct_1 / print_batch_counter,
                                                      100. * correct1 / (args.val_batch_size * (batch_idx + 1)),
                                                      100. * print_correct_5 / print_batch_counter,
                                                      100. * correct5 / (args.val_batch_size * (batch_idx + 1))))
            print_correct_1, print_correct_5 = 0 , 0
            print_batch_counter = 0


    return loss.item(), correct1 / len(data_loader.dataset), correct5 / len(data_loader.dataset)


def train(data_loader, model, criterion, device, epoch, optimizer, scheduler,
          training_stats_logger=None, num_of_batches=None):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, device, epoch, training=True, optimizer=optimizer,
                   scheduler=scheduler, training_stats_logger=training_stats_logger,num_of_batches=num_of_batches)


def validate(data_loader, model, criterion, device, epoch, num_of_batches=None, stats_phase=False):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, device, epoch, num_of_batches=num_of_batches,
                   training=False, optimizer=None, scheduler=None, stats_phase=stats_phase)


# TODO: separate file
def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def plot_bn_statistic(model):
    # plot histogram
    i = 0
    for m in model.module.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    gaussian_numbers = m._parameters[p].view(-1).cpu().detach().numpy()
                    plt.hist(gaussian_numbers, bins=256)
                    file_name = p + '_layer_' + str(i)
                    directory = './plot_stats'
                    if not os.path.isdir(directory):
                        os.mkdir(directory)

                    file_name = os.path.join(directory, file_name + '.png')
                    plt.savefig(file_name)
                    plt.close()

            for b in m._buffers:
                if m._buffers[b] is not None:
                    gaussian_numbers = m._buffers[b].view(-1).cpu().detach().numpy()
                    plt.hist(gaussian_numbers, bins=256)
                    file_name = b + '_layer_' + str(i)
                    directory = './plot_stats'
                    if not os.path.isdir(directory):
                        os.mkdir(directory)

                    file_name = os.path.join(directory, file_name + '.png')
                    plt.savefig(file_name)
                    plt.close()
            i += 1

def migrate_models(model, target_model, best_epoch, model_name='marvis_mobilenet_multi_gpu'):
    """
    This code snnipet is meant to adapt pre-trained model to a new model containing buffers
    """
    module_list = [m for m in list(model.modules()) if isinstance(m, torch.nn.Conv2d) or
                   isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm2d)]
    if args.gpus is not None:
        target_model = torch.nn.DataParallel(target_model, args.gpus)

    target_module_list = [m for m in list(target_model.modules()) if isinstance(m, torch.nn.Conv2d) or
                   isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm2d)]
    for idx, m in enumerate(module_list):
        for p in m._parameters:
            if m._parameters[p] is not None:
                target_module_list[idx]._parameters[p].data = m._parameters[p].data.clone()

        for b in m._buffers:  # For batchnorm stats
            if m._buffers[b] is not None:
                    target_module_list[idx]._buffers[b].data = m._buffers[b].data.clone()

    save_dir = os.path.join('./trained_models', model_name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_checkpoint({
        'epoch': best_epoch,
        'model': args.model,
        'config': args.model_config,
        'state_dict': target_model.state_dict(),
        'best_prec1': best_epoch
    }, True, path=save_dir)


def gather_clamp_statistic(model):
    act_layer_num = 0
    conv_linear_layer_num = 0

    # Activation clamp  are taken from the model itself
    for layer in list(model.modules()):
        if isinstance(layer, actquant.ActQuantBuffers):
            layer_name = 'Activation_{}_clamp_val'.format(act_layer_num)
            if layer.clamp_val.data is not None:
                if layer_name not in clamp_stats_dict:
                    clamp_stats_dict[layer_name] = []
                    clamp_stats_dict[layer_name].append(layer.clamp_val.data.item())
                else:
                    clamp_stats_dict[layer_name].append(layer.clamp_val.data.item())
            act_layer_num += 1

    for layer in list(model.modules()):
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            for p in layer._parameters:
                if layer._parameters[p] is not None:
                    if p == 'layer_basis':
                        layer_name = 'Conv_Fc_{}_clamp_val'.format(conv_linear_layer_num)
                        lb = layer._parameters[p]
                        if lb is not None:
                            clamp_val = (2 ** (args.bitwidth - 1) - 1) * lb * layer.layer_b
                            if layer_name not in clamp_stats_dict:
                                clamp_stats_dict[layer_name] = []
                                clamp_stats_dict[layer_name].append(clamp_val.item())
                            else:
                                clamp_stats_dict[layer_name].append(clamp_val.item())
                        conv_linear_layer_num += 1

def plot_clamp_statistic(stats_dict, save_path):
    # plot histogram
    for k, v in stats_dict.items():
        epoch = len(stats_dict[k])
        plt.plot(list(range(epoch)), v,'.')
        file_name = k
        directory = os.path.join(save_path, 'clamp_plot_stats')
        # directory = 'clamp_plot_stats'
        if not os.path.isdir(directory):
            os.mkdir(directory)

        file_name = os.path.join(directory, file_name + '.png')
        plt.savefig(file_name)
        plt.close()


def plot_clamp_statistic_from_file(dict_file, act_layers_list,  save_path):
    plt.figure()
    file_name = os.path.join(save_path,'unified_activation_clamp.png')

    stats_dict = np.load(dict_file)
    dict_keys = list(stats_dict.item().keys())
    for layer in act_layers_list:
        act_vals = stats_dict.item()[dict_keys[layer]]
        epoch = len(act_vals)
        plt.plot(list(range(epoch)), act_vals)
        plt.xlabel('epoch')
        plt.ylabel('Clamp Value')
        plt.savefig(file_name)

    plt.show()



def plot_cos_loss(stats_dict, save_path):

    for k, v in stats_dict.items():
        epoch = len(stats_dict[k])
        plt.plot(list(range(epoch)), v,'.')
        file_name = k
        directory = os.path.join(save_path, 'cos_loss')
        if not os.path.isdir(directory):
            os.mkdir(directory)

        file_name = os.path.join(directory, file_name + '.png')
        plt.savefig(file_name)
        plt.close()

def gather_cos_loss(model):
    num_layers = len(model.module.quantize.cosine_sim_loss)
    total_cosine_loss=0
    layer_num = 0
    for layer, cos_loss in model.module.quantize.cosine_sim_loss.items():
        total_cosine_loss += cos_loss
        layer_string = "cos_loss_layer_{}".format(layer_num)

        if layer_string not in cos_loss_dict:
            cos_loss_dict[layer_string] = []
            cos_loss_dict[layer_string].append(cos_loss)
        else:
            cos_loss_dict[layer_string].append(cos_loss)
        layer_num += 1

    if 'total_cosine_loss' not in cos_loss_dict:
        cos_loss_dict['total_cosine_loss'] = []
        cos_loss_dict['total_cosine_loss'].append(total_cosine_loss/num_layers)
    else:
        cos_loss_dict['total_cosine_loss'].append(total_cosine_loss/num_layers)
    return


def plot_act_quant_error_statistic(model, save_path):
    for layer in model.module.modules():
        # if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        if isinstance(layer, actquant.ActQuantBuffers):
            i = layer.layer_num
            plt.hist(layer.quant_error, bins=256)
            file_name = 'layer_' + str(i)
            directory = os.path.join(save_path, 'act_quant_error_stats')
            if not os.path.isdir(directory):
                os.mkdir(directory)
            file_name = os.path.join(directory, file_name + '.png')
            plt.savefig(file_name)
            plt.close()
    return


def plot_weight_quant_error_statistic(model, save_path):
    i = 0
    for layer, stats in model.module.quantize.quant_error.items():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            plt.hist(np.concatenate(stats), bins=256)
            file_name = 'layer_' + str(i)
            directory = os.path.join(save_path, 'weight_quant_error_stats')
            if not os.path.isdir(directory):
                os.mkdir(directory)
            full_path = os.path.join(directory, file_name + '.png')
            plt.savefig(full_path)
            plt.close()
            i += 1
    return

if __name__ == '__main__':
    main()

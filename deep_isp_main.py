import argparse
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
import time
from models.deep_isp_model import DenoisingNet
from msr_demosaic import MSRDemosaic
import deep_isp_utils as utils
from collections import OrderedDict
import shutil
import matplotlib.pyplot as plt
from loss import *
from datetime import datetime

import numpy as np
from torch import nn
import quantize
import actquant

DATA_PATH = os.path.join('data','datasets','MSR-Demosaicing')
GPUS_DEFAULT = '0' if torch.cuda.is_available() else None
OUTPUT_DIR = os.path.join('.','output')

if (GPUS_DEFAULT is not None):
    print ("running with gpu")
else:
    print ("running with cpu")

#recommanded cmd: --batch_size=1 --num_denoise_layers=20 --num_workers=0 --start-epoch=6000 --resume=output\pretrained\best_checkpoint.pth.tar --quant=True

parser = argparse.ArgumentParser(description='Denoising training with PyTorch')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='random seed')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16, help='Number of epochs to train.')
parser.add_argument('--num_denoise_layers', type=int, default=20, help='num of layers.')
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='The learning rate.')
parser.add_argument('--decay', '-d', type=float, default=0, help='Weight decay (L2 penalty).')
parser.add_argument('--gpus', default=GPUS_DEFAULT, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('--datapath', type=str, default=DATA_PATH, help='Path to MSR-Demosaicing dataset')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help='Path to save model and results')

parser.add_argument('--quant_epoch_step', type=int, default=50, help='quant_bitwidth.')
parser.add_argument('--num_workers', type=int, default=4, help='Num of workers for data.')

parser.add_argument('--quant_start_stage', type=int, default=0, help='Num of workers for data.')

parser.add_argument('--inject_noise', default=False, type=lambda x: (str(x).lower() == 'true'), help='use preproccesing for the grad')
parser.add_argument('--show_test_result', type=lambda x: (str(x).lower() == 'true'), default=False, help='show figures of test result')
parser.add_argument('--quant', default=False, type=lambda x: (str(x).lower() == 'true') , help='use preproccesing for the grad')
parser.add_argument('--quant_bitwidth', type=int, default=32, help='quant_bitwidth.')

parser.add_argument('--inject_act_noise', default=False, type=lambda x: (str(x).lower() == 'true'), help='use preproccesing for the grad')
parser.add_argument('--act_quant', default=False, type=lambda x: (str(x).lower() == 'true') , help='use preproccesing for the grad')
parser.add_argument('--act_bitwidth', type=int, default=32, help='quant_bitwidth.')
parser.add_argument('--step', type=int, default=19, help='amount of split the layer in quant.')

parser.add_argument('--set_gpu', type=lambda x: (str(x).lower() == 'true'), default=False, help='show figures of test result')
parser.add_argument('--adaptive_lr', type=lambda x: (str(x).lower() == 'true'), default=True, help='show figures of test result')

parser.add_argument('--enable_decay', type=lambda x: (str(x).lower() == 'true'), default=False, help='decay_enable')
parser.add_argument('--weight_relu', type=lambda x: (str(x).lower() == 'true'), default=False, help='weight_relu')
parser.add_argument('--weight_grad_after_quant', type=lambda x: (str(x).lower() == 'true'), default=False, help='weight_grad_after_quant')
parser.add_argument('--random_inject_noise', type=lambda x: (str(x).lower() == 'true'), default=False, help='random_inject_noise')

parser.add_argument('--stage_only_clamp', type=lambda x: (str(x).lower() == 'true'), default=False, help='stage_only_clamp')
parser.add_argument('--wrpn', type=lambda x: (str(x).lower() == 'true'), default=False, help='wrpn quantization')

parser.add_argument('--copy_statistics', type=lambda x: (str(x).lower() == 'true'), default=True, help='copy_statistics')

parser.add_argument('--quant_decay', type=float, default=0.0005, help='quant decay.')

parser.add_argument('--val_part', type=float, default=0.1, help='quant decay.')

args = parser.parse_args()

transformation = utils.JointCompose([
    utils.JointHorizontalFlip(),
    utils.JointVerticalFlip(),
    #utils.JointNormailze(means = [0.485,0.456,0.406],stds = [1,1,1]), #TODO consider use
    utils.JointToTensor(),
])
val_transformation = utils.JointCompose([
    #utils.JointNormailze(means = [0.485,0.456,0.406],stds = [1,1,1]),
    utils.JointToTensor(),
])

VAL_PART = args.val_part

trainset = MSRDemosaic(root=args.datapath, train=True, validation_part=VAL_PART, transform=transformation)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

statistic_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=args.num_workers)

valset = MSRDemosaic(root=args.datapath, train=False, validation_part=VAL_PART, validation=True, transform=val_transformation)
val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=args.num_workers)

testset = MSRDemosaic(root=args.datapath, train=False, transform=val_transformation)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def load_model(model,checkpoint):

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k[0:6] == 'module.' else k # remove `module. if needed (happen when the model created with DataParallel
        #new_state_dict[name] = v
        new_state_dict[name] = v if v.dim() > 1 or 'num_batches_tracked' in name else v*v.new_ones(1)

    # load params
    model.load_state_dict(new_state_dict, strict=False) #strict false in case the loaded doesn't have alll variables like running mean

    # if 'layers_b_dict' in checkpoint:
    #     model.layers_b_dict = checkpoint['layers_b_dict']

    # new_state_dict_with_pointers = OrderedDict()
    # for key in  model.state_dict().keys() :
    #    if key in new_state_dict:
    #        new_state_dict_with_pointers[key] = new_state_dict[key]
    #    else:
    #        new_state_dict_with_pointers[key] = model.state_dict()[key]
    # model.load_state_dict(new_state_dict_with_pointers)


def check_if_need_to_collect_statistics(model):
    for layer in model.modules():
        if isinstance(layer, actquant.ActQuantBuffers):
            if hasattr(layer, 'running_std') and float(layer.running_std) != 0:
                return False

    return True

def adjust_parameters(model):
    modules_list = list(model.modules())
    layers_to_change = [x for x in modules_list if (isinstance(x, nn.Conv2d) and x.out_channels == 3)]
    for m in layers_to_change:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
            for p in m._parameters:
                    m._parameters[p].data = m._parameters[p].data/ 10 * 8


def main():
    if args.gpus is not None:
        torch.cuda.manual_seed_all(args.seed)
        #args.gpus = [int(i) for i in args.gpus.split(',')]
        #torch.cuda.set_device(args.gpus[0])

    if args.set_gpu :
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])

    model = DenoisingNet(num_denoise_layers=args.num_denoise_layers, quant=args.quant , noise=args.inject_noise, bitwidth=args.quant_bitwidth, quant_epoch_step=args.quant_epoch_step,
                         act_noise=args.inject_act_noise , act_bitwidth= args.act_bitwidth , act_quant=args.act_quant, use_cuda=(args.gpus is not None), quant_start_stage=args.quant_start_stage,
                         weight_relu=args.weight_relu, weight_grad_after_quant=args.weight_grad_after_quant, random_inject_noise = args.random_inject_noise
                         , step=args.step, wrpn=args.wrpn)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(model)
    print('number of parameters: {}'.format(num_parameters))

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.out_dir = os.path.join(args.out_dir, time_stamp)

    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    checkpoint_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    csv_path = os.path.join(args.out_dir, 'training_stats.csv')

    if args.gpus is not None:
        model.cuda()
        device = 'cuda:' + str(args.gpus[0])

        torch.cuda.set_device(args.gpus[0])

    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss()

    if args.resume:
        checkpoint_file = args.resume
        if os.path.isfile(checkpoint_file):
            print("loading checkpoint {}".format(args.resume))
            if args.gpus is not None:
                checkpoint = torch.load(checkpoint_file, map_location=device)
            else:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')

            load_model(model,checkpoint)
            #adjust_parameters(model)
            optimizer = checkpoint['optim']
            #torch.save({'state_dict': model.state_dict(), 'epoch': checkpoint['epoch'], 'optim': optimizer}, checkpoint_path)

            print("loaded checkpoint {} (epoch {})".format(checkpoint_file, checkpoint['epoch']))
        else:
            print("no checkpoint found at {}".format(args.resume))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate, weight_decay=args.decay) #in case i want to start with same layers with no change
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate, weight_decay=args.decay,momentum=0)  # in case i want to start with same layers with no change

    model_pointer = model.module if args.gpus and len(args.gpus) > 1 else model    # model for not dataparallel model._modules['module']

    if check_if_need_to_collect_statistics(model):

        for layer in model.modules():
            if isinstance(layer, actquant.ActQuantBuffers):
                layer.pre_training_statistics = True  # Turn on pre-training activation statistics calculation
        model_pointer.statistics_phase = True

        collect_statistic( model, statistic_loader)  # Run validation on training set for statistics
        model_pointer.quantize.get_act_max_value_from_pre_calc_stats(list(model.modules()))
        _ = model_pointer.quantize.set_weight_basis(list(model.modules()), None)

        for layer in model.modules():
            if isinstance(layer, actquant.ActQuantBuffers):
                layer.pre_training_statistics = False  # Turn off pre-training activation statistics calculation
        model_pointer.statistics_phase = False

    else:  # Maximal activation values still need to be derived from loaded stats
        model_pointer.quantize.get_act_max_value_from_pre_calc_stats(list(model.modules()))


    if args.stage_only_clamp:
        model.only_clamp = True
        for epoch in range(0, 200):
            train_loss = train(model, epoch, optimizer, criterion)
            print('train loss: clamp check {:.3e}'.format(train_loss))

        torch.save({'state_dict': model.state_dict(), 'epoch': checkpoint['epoch'], 'optim': optimizer}, checkpoint_path)

        model.only_clamp = False
    ########

    # fast forward to curr stage
    for i in range(args.quant_start_stage):
        model.switch_stage(0)

    if args.start_epoch == 0:
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_psnr,decay_loss,dur\n')

    best_psnr = 0
    for epoch in tqdm(range(args.start_epoch,args.epochs), initial=args.start_epoch):
        t = time.time()
        train_loss = train(model, epoch, optimizer, criterion)
        test_loss, test_psnr , decay_loss = test(model, criterion)
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optim': optimizer}, checkpoint_path)

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            shutil.copy(checkpoint_path, os.path.join(args.out_dir, 'best_checkpoint.pth.tar'))

        dur = time.time() - t
        tqdm.write('\nTrain loss: {:.3e}, Val loss: {:.3e}, Val PSNR: {:.3f}, Decay Loss: {:.3f}, Duration: {}\n'.format(train_loss, test_loss,test_psnr, decay_loss, dur))

        with open(csv_path, 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(epoch, train_loss, test_loss,test_psnr, decay_loss, dur))

        if epoch % 20 == 0:
            for layer in model.modules():
                if isinstance(layer, actquant.ActQuantBuffers):
                    layer.print_clamp()

        plot_weight_quant_error_statistic(model, args.out_dir)


    print("Evaluating test set:\n")
    test_loss, test_psnr , decay_loss = test(model, criterion, on_test_set=True)
    print('\nTest loss: {:.3e}, Test PSNR: {:.3f}'.format(test_loss, test_psnr))



def collect_statistic(model, statistic_loader):
    model.eval()

    for batch_idx, (data, target, _) in enumerate(tqdm(statistic_loader)):
        if args.gpus is not None:
            data, target = data.cuda(async=True), target.cuda(async=True)

        model(data)

def train(model, epoch, optimizer, criterion):
    model.train()
    train_loss = 0

    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        if args.gpus is not None:
            data, target = data.cuda(async=True), target.cuda(async=True)

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        epoch_progress = epoch + batch_idx / len(train_loader)
        #model.module.switch_stage(epoch_progress)
        model_pointer =  model.module if args.gpus and len(args.gpus) > 1 else model
        model_pointer.switch_stage(epoch_progress)
        #model.set_train_epoch(epoch)

        # torch.cuda.synchronize()
        # a = time.perf_counter()

        output = model(data)

        # torch.cuda.synchronize() # wait for mm to finish
        # b = time.perf_counter()
        # print('batch GPU {:.03e}s'.format(b - a))
        # break


        #loss = criterion(output, target)
        loss_for_psnr, loss , weight_decay_loss = calc_loss(output, target, criterion, model,args)
        loss.backward()
        optimizer.step()
        train_loss += output.shape[0] * loss_for_psnr.item()  # sum up batch loss

    train_loss /= len(train_loader.dataset)
    return train_loss

def unbias_image(img):
    return  torch.clamp(img, -0.5 , 0.5).data.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 0.5  #the clamp is becuase the value should be between 0-1


def plot_images(output,target, input,test_index):
    plt.figure()
    output = unbias_image(output)
    target = unbias_image(target)
    input = unbias_image(input)

    image_row , image_col = output.shape[0], output.shape[1]

    #save the images
    plot_all = False
    if (plot_all):
        num_of_images = 3
        figure = np.zeros((image_row , image_col * num_of_images, 3  ))
        figure[:, 0 * image_col: image_col * 1] = input
        figure[:, 1 * image_col: image_col * 2] = target
        figure[:, 2 * image_col: image_col * 3] = output
    else:
        figure = output


    plt.imshow(figure, interpolation='nearest')
    #plt.show()

    file_name = 'test_image_' + str(test_index) + '.png'
    plt.savefig(file_name) #,dpi=400
    #plt.close()



def test(model, criterion, on_test_set=False):
    model.eval()
    test_loss = 0
    psnr = 0
    if on_test_set:
        loader = test_loader
    else:
        loader = val_loader

    for batch_idx, (data, target, fname) in enumerate(tqdm(loader)):
        if args.gpus is not None:
            data, target = data.cuda(async=True), target.cuda(async=True)

        with torch.no_grad():
            data, target = Variable(data), Variable(target)

        output = model(data)
        #cur_loss = criterion(output, target).data[0]
        loss_for_psnr, loss , weight_decay_loss = calc_loss(output, target, criterion, model,args)
        test_loss += output.shape[0] * loss.item() # sum up batch loss
        psnr += utils.mse2psnr(loss_for_psnr.item() ) # true only for batch_size == 1



        if (args.show_test_result):
            plot_images(output,target, data, batch_idx)



    test_loss /= len(loader.dataset)
    psnr /= len(loader.dataset)

    return test_loss, psnr , weight_decay_loss.data[0]



def plot_weight_quant_error_statistic(model, save_path):
    i = 0
    for layer, stats in model.quantize.quant_error.items():
        # if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        if isinstance(layer, torch.nn.Conv2d):
            # gaussian_numbers = layer.quant_error.view(-1).cpu().detach().numpy()
            plt.hist(np.concatenate(stats).ravel(), bins=256)
            file_name = 'layer_' + str(i)
            directory = os.path.join(save_path, 'weight_quant_error_stats')

            if not os.path.isdir(directory):
                os.mkdir(directory)

            plt.title('Quantization Error Distribution')
            plt.xlabel('Q(W) - W')
            # plt.ylabel('Clamp Value')

            file_name = os.path.join(directory, file_name + '.png')
            plt.savefig(file_name)
            plt.close()
            i += 1
    return

if __name__ == '__main__':
    main()

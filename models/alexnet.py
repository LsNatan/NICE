import torch.nn as nn
import torchvision.transforms as transforms
import quantize
import torch
import numpy as np
import math

__all__ = ['alexnet']

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_()  # The original initialization in class _BatchNorm
            m.bias.data.zero_()       # The original initialization in class _BatchNorm       

            # m.weight.data.fill_(1)
            # m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features
            m.weight.data.normal_(0, math.sqrt(2. / n))
                       

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000,quant=False, noise=False, bitwidth=32,step = 2):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        self.quant=quant
        self.noise=noise
        self.bitwidth=bitwidth      
        self.training_stage=0
        self.step = step

        init_model(self)
        print(self.quant,self.noise,self.bitwidth)

        self.regime = [
            # {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
            #  'weight_decay': 1e-4, 'momentum': 0.9},

            # {'epoch': 25, 'momentum': 0.9},

            # {'epoch': 30, 'lr': 1e-2},
            # {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            # {'epoch': 90, 'lr': 1e-4}

            {'epoch': 1, 'optimizer': 'SGD', 'lr': 1e-2,
             'weight_decay': 1e-4, 'momentum': 0.9},

            {'epoch': 3, 'momentum': 0.9},

            {'epoch': 5, 'lr': 1e-2},
            {'epoch': 7, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 9, 'lr': 1e-4}

        ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

        modules_list = list(self.modules())
        self.layers_list = [x for x in modules_list if  isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear) ]

        # chunk = len(self.layers_list)//self.step
        # self.layers_half_one=self.layers_list[:chunk]
        # self.layers_half_two=self.layers_list[chunk:]
        self.layers_steps = np.array_split(self.layers_list, self.step)


    def switch_stage(self,current_step):
        # self.training_stage = 1
        # for layer in self.layers_half_one:
        #     for param in layer.parameters():
        #         param.requires_grad = False

        self.training_stage = current_step
        for step in self.layers_steps[:self.training_stage]:
            for layer in step:
                for param in layer.parameters():
                    param.requires_grad = False

    def print_max_min_params(self):          
        weight_max = 0
        weight_min = 1e8
        bias_max = 0
        bias_min = 1e8

        for m in self.modules():
            if ( isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
                if torch.max(torch.abs(m.weight.data)) > weight_max:
                    weight_max = torch.max(torch.abs(m.weight.data))

                if torch.min(torch.abs(m.weight.data)) < weight_min:
                    weight_min = torch.min(torch.abs(m.weight.data)) 

                if m.bias is not None:
                    if torch.max(torch.abs(m.bias.data)) > bias_max:
                        bias_max = torch.max(torch.abs(m.bias.data))

                    if torch.min(torch.abs(m.bias.data)) < bias_min:
                        bias_min = torch.min(torch.abs(m.bias.data)) 


        print("max weight is : {}".format(weight_max))
        print("min weight is : {}".format(weight_min))
        if m.bias is not None:        
            print("max bias is : {}".format(bias_max))
            print("min bias is : {}".format(bias_min))


    def forward(self, x):
        temp_saved = {}

        if self.quant and not self.training:
            temp_saved = quantize.backup_weights(self.layers_list,{})
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
                temp_saved = quantize.backup_weights(self.layers_steps[i],temp_saved)
                quantize.quantize(self.layers_steps[i], bitwidth=self.bitwidth)
                

    
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)

        if self.quant and not self.training:
            quantize.restore_weights(self.layers_list,temp_saved)

        # elif self.noise and self.training:
        #     if self.training_stage==0:
        #         quantize.restore_weights(self.layers_half_one, temp_saved)
        #     else:
        #         quantize.restore_weights(self.layers_half_one+self.layers_half_two, temp_saved)

        elif self.noise and self.training:
            quantize.restore_weights(self.layers_steps[self.training_stage], temp_saved) #Restore the noised layers
            for i in range(self.training_stage):
                quantize.restore_weights(self.layers_steps[i], temp_saved) #Restore the quantized layers

        
        return x


def alexnet(**kwargs):
    num_classes, quantize, noise, bitwidth,step = map( kwargs.get, ['num_classes','quantize','noise','bitwidth','step'])

    num_classes = getattr(kwargs, 'num_classes', 1000)
    # quantize = getattr(kwargs, 'quantize', True)
    # noise = getattr(kwargs, 'noise', True)
    # bitwidth = getattr(kwargs, 'bitwidth', 7)
    #print(kwargs)

    return AlexNetOWT_BN(num_classes,quant=quantize, noise=noise, bitwidth=bitwidth,step = step)

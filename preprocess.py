import torch
import torchvision.transforms as transforms
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats, integer_values=True, norm=True):
    if integer_values:
        print("Quantized Validation Loader")

        t_list = [
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.floor(255 * (x - 0.5)))
            ]
    else:
        print("Full precision Validation Loader")

        t_list = [
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ]
        if norm:
            t_list += [transforms.Normalize(**normalize)]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats, integer_values=True):
    padding = int((scale_size - input_size) / 2)
    if integer_values:
        print("Quantized Training Loader")

        return transforms.Compose([
            # transforms.RandomCrop(input_size, padding=padding),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),# According to "original" environment
            transforms.RandomCrop(input_size, padding=padding),# According to "original" environment
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.floor(255 * (x - 0.5)))
            ])
    else:
        print("Full Precision Training Loader")

        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize),
            ])


def inception_preproccess(input_size, normalize=__imagenet_stats, integer_values=True, norm=True):
    if integer_values:
        print("Quantized Training Loader")
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalization with shifting to [0,1] range
            transforms.Normalize(**normalize),
            transforms.Lambda(lambda x: x.sub_(x.min())),
            transforms.Lambda(lambda x: x.mul_(1/x.max())),

            # transforms.Lambda(lambda x: x[0].sub_(x[0].min())),
            # transforms.Lambda(lambda x: x[1].sub_(x[1].min())),
            # transforms.Lambda(lambda x: x[2].sub_(x[2].min())),
            #
            # transforms.Lambda(lambda x: x[0].mul(1/x[0])),
            # transforms.Lambda(lambda x: x[1].mul(1/x[1])),
            # transforms.Lambda(lambda x: x[2].mul(1/x[2])),
            transforms.Lambda(lambda x: torch.floor(255 * (x - 0.5)))
        ])
    else:
        print("Full Precision Training Loader")
        t_list = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        if norm:
            t_list += [transforms.Normalize(**normalize)]
        return transforms.Compose(t_list)

def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True, integer_values=True, norm=True):
    normalize = normalize or __imagenet_stats
    if name == 'imagenet':
        scale_size = scale_size or 256
        input_size = input_size or 224
        if augment:
            return inception_preproccess(input_size, normalize=normalize, integer_values=integer_values,
                                         norm=norm)
        else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize, integer_values=integer_values,
                              norm=norm)
    elif 'cifar' in name:
        input_size = input_size or 32
        if augment:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize, integer_values=integer_values)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize, integer_values=integer_values)
    elif name == 'mnist':
        normalize = {'mean': [0.5], 'std': [0.5]}
        input_size = input_size or 28
        if augment:
            scale_size = scale_size or 32
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

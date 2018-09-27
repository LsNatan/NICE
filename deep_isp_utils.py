import random

import numpy as np
import torch


# from torchvision.transforms import functional as F

def mosaic_then_demosaic(rgb, pattern='grbg'):
    from scipy.ndimage.filters import convolve
    rgb = np.squeeze(rgb)
    if rgb.ndim == 2:  # if input is in CFA format, same code should work with just stacking the input
        rgb = np.stack([rgb] * 3)

    in_shape = rgb.shape
    if in_shape[2] == 3:
        rgb = np.transpose(rgb, [2, 0, 1])

    mask = np.zeros_like(rgb)
    if pattern == 'grbg':
        mask[0, 0::2, 1::2] = 1  # r
        mask[1, 0::2, 0::2] = 1  # g1
        mask[1, 1::2, 1::2] = 1  # g2
        mask[2, 1::2, 0::2] = 1  # b
    elif pattern == 'rggb':
        mask[0, 0::2, 0::2] = 1  # r
        mask[1, 0::2, 1::2] = 1  # g1
        mask[1, 1::2, 0::2] = 1  # g2
        mask[2, 1::2, 1::2] = 1  # b
    else:
        raise NotImplementedError

    H_G = np.asarray(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4  # yapf: disable

    H_RB = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    rgb[0, ...] = convolve(rgb[0, ...] * mask[0, ...], H_RB, mode='mirror')
    rgb[1, ...] = convolve(rgb[1, ...] * mask[1, ...], H_G, mode='mirror')
    rgb[2, ...] = convolve(rgb[2, ...] * mask[2, ...], H_RB, mode='mirror')

    if in_shape[2] == 3:
        rgb = np.transpose(rgb, [1, 2, 0])
    return rgb


def mse2psnr(mse, max_val=1.0):
    return 10.0 * np.log10(max_val / mse)


class JointHorizontalFlip(object):
    """Horizontally flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        if random.random() < 0.5:
            return img[:,:,::-1].copy(), target[:,:,::-1].copy() #F.hflip(img), F.hflip(target)
        return img, target

class JointVerticalFlip(object):
    """Vertically flip the given pair of PIL Images randomly with a probability of 0.5."""

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        if random.random() < 0.5:
            return img[:,::-1,:].copy(), target[:,::-1,:].copy() # F.vflip(img), F.vflip(target)
        return img, target

class JointNormailze(object):
    """Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]"""
    def __init__(self, means, stds):
        self.means, self.stds = means, stds


    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        img -= np.array(self.means)[:,None,None]
        img /= np.array(self.stds)[:,None,None]

        target -= np.array(self.means)[:,None,None]
        target /= np.array(self.stds)[:,None,None]

        return img, target


class JointUnNormailze(object):
    """Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]"""
    def __init__(self, means, stds):
        self.means, self.stds = means, stds


    def __call__(self, img,target):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Image to be flipped.

        Returns:
            PIL Image, PIL Image: Randomly flipped images.
        """
        img += np.array(self.means)[:,None,None]
        img *= np.array(self.stds)[:,None,None]

        target += np.array(self.means)[:,None,None]
        target *= np.array(self.stds)[:,None,None]

        return img

class JointCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class JointToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic1, pic2):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic1), to_tensor(pic2)

def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    # handle numpy array
    img = torch.from_numpy(pic)
    # HACK
    return img.float()

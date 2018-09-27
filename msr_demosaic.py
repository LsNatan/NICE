import os
import os.path

import numpy as np
import torch.utils.data as data
# from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import deep_isp_utils as utils


class MSRDemosaic(data.Dataset):
    """`MSR Demosaicing <https://www.microsoft.com/en-us/download/details.aspx?id=52535>`_  Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Dataset_LINEAR_with_noise`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a pair of PIL images
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """
    base_folder = os.path.join('Dataset_LINEAR_with_noise', 'bayer_panasonic')
    train_list = ['train.txt', 'validation.txt']

    test_list = ['test.txt']

    def read_pair_imgs(self, id):
        gt = plt.imread(os.path.join(self.dir_path, 'groundtruth', id + '.png'))[:, :, :3] - 0.5 # to check the mean ~0.17, std ~0.104   ,np.mean( np.resize( np.transpose(  plt.imread(os.path.join(self.dir_path, 'groundtruth', id + '.png'))[:, :, :3]  , (2,0,1) ) ,(3,290040) ) , axis=1 )
        gt = np.transpose(gt, (2, 0, 1))

        mosaiced = plt.imread(os.path.join(self.dir_path, 'input', id + '.png'))
        image = utils.mosaic_then_demosaic(mosaiced, 'rggb') - 0.5
        return image, gt

    def __init__(self, root, train=True, validation=False, validation_part=0.1, transform=None):
        assert (not (train and validation))
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation  # validation set
        self.dir_path = os.path.join(self.root, self.base_folder)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.')

        # now load the picked numpy arrays
        if self.train or self.validation:

            self.train_data = []
            self.train_labels = []

            for f in self.train_list:
                file = os.path.join(self.dir_path, f)
                for line in open(file, 'r'):
                    line = line.rstrip()
                    im, gt = self.read_pair_imgs(line)
                    self.train_data.append(im)
                    self.train_labels.append(gt)

            self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
                                                                                    self.train_data,
                                                                                    self.train_labels,
                                                                                    test_size=validation_part,
                                                                                    random_state=32)

        else:
            self.test_data = []
            self.test_labels = []
            self.test_filenames = []
            for f in self.test_list:
                file = os.path.join(self.dir_path, f)
                for line in open(file, 'r'):
                    line = line.rstrip()
                    im, gt = self.read_pair_imgs(line)
                    self.test_data.append(im)   # in order to view images plt.imshow(gt.transpose(1, 2, 0)) ,plt.savefig('yoav.png')
                    self.test_labels.append(gt)
                    self.test_filenames.append(line)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            filename = 0
        elif self.validation:
            img, target = self.val_data[index], self.val_labels[index]
            filename = 0
        else:
            img, target, filename = self.test_data[index], self.test_labels[index], self.test_filenames[index]

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, filename

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.validation:
            return len(self.val_data)
        else:
            return len(self.test_data)

import os
import types
import numpy as np
import torch
import torchvision
from os.path import join
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils.utils import ignored

__all__ = ['get_data_loaders', 'get_data_loaders_no_ssl']

class X_Ray_Images(Dataset):

    def __init__(self, base_dir, transform, train = True,
                       balanced = True, *args, **kwargs):

        set_type = 'train' if train else 'test'
        balanced_string = '' if balanced else '_unbalanced'

        self.transform = transform
        self.X = np.load(join(base_dir, \
                    f'x_ray_images/xray_x_{set_type}{balanced_string}.npy'))
        self.y = np.load(join(base_dir, \
                    f'x_ray_images/xray_y_{set_type}{balanced_string}.npy'))

        self.y = torch.from_numpy(self.y).long()
        assert self.y.shape[0] == self.X.shape[0], \
                f'Dim 0 of {set_type} images and labels must be same size!'

        # colors must be last dimension, necessary for PIL transformations
        assert self.X.shape[-1] == 3, \
                'last dimension must be colors.'

        # original labels go from 1 - 7, pytorch needs 0 - 6
        self.y -= 1
        assert int(self.y.max()) == 6 and int(self.y.min()) == 0, \
                'class labels must be between 0 and 6.'


    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]

         ## convert to PIL-image
        img = Image.fromarray(data)

        #apply the transformations and return tensors
        return self.transform(img), label


    def __len__(self):
        return self.X.shape[0]


class Labeled_Dataset(Dataset):

    def __init__(self, args,
                 transform = None,
                 indexs = None,
                 target_transform = None,
                 train = True, **kwargs):
        super(Labeled_Dataset, self).__init__()

        self.transform = transform
        self.target_transform = target_transform

        if args.dataset == 'cifar':
            ds = torchvision.datasets.CIFAR10(root = args.data_dir,
                                              train = train,
                                              download = True)
            self.X = ds.data
            self.y = np.array(ds.targets)

            # equals np.mean(train_set.train_data, axis=(0,1,2))/255
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616)

        elif args.dataset == 'x_ray':
            ds = X_Ray_Images(base_dir = args.data_dir,
                              transform = transform,
                              train = train,
                              balanced = True)
            self.X = ds.X
            self.y = ds.y
            self.mean = (0.4881445, 0.4881445, 0.4881445)
            self.std  = (0.24949823, 0.24949823, 0.24949823)

        if indexs is not None:
            self.X = self.X[indexs]
            self.y = self.y[indexs]

        self.X = transpose(normalise(self.X, mean=self.mean, std=self.std))

    def __getitem__(self, index):

        img, target = self.X[index], self.y[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            with ignored(TypeError):
                target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.X.shape[0]


class Unlabeled_Dataset(Labeled_Dataset):

    def __init__(self, args, indexs, transform,
                 target_transform=None,
                 train = True, **kwargs):
        super(Unlabeled_Dataset, self).__init__(args,
                                indexs = indexs,
                                transform=transform,
                                target_transform=target_transform,
                                train = train, **kwargs)
        self.y = np.array([-1 for i in range(len(self.y))])


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):

        x = pad(x, 4)
        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


def get_data_loaders(args):

    # get datasets
    print(f'==> Preparing {args.dataset}')

    if args.dataset == 'cifar':
        crop_size = 32
        args.num_classes = 10
    elif args.dataset == 'x_ray':
        crop_size = 224
        args.num_classes = 7

    # transformations
    transform_train = transforms.Compose([
                        RandomPadandCrop(crop_size),
                        RandomFlip(),
                        ToTensor()])

    transform_val = transforms.Compose([
                        ToTensor()])

    train_labeled_set, train_unlabeled_set, \
    val_set, test_set = get_datasets(transform_train = transform_train,
                                     transform_val = transform_val,
                                     args = args)

    labeled_trainloader = DataLoader(train_labeled_set,
                                          batch_size = args.batch_size,
                                          shuffle = True,
                                          num_workers = 0,
                                          drop_last = True)
    unlabeled_trainloader = DataLoader(train_unlabeled_set,
                                          batch_size = args.batch_size,
                                          shuffle = True,
                                          num_workers = 0,
                                          drop_last = True)
    val_loader = DataLoader(val_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=0)
    test_loader = DataLoader(test_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=0)
    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, args


def get_datasets(transform_train, transform_val, args):

    if args.dataset == 'cifar':
        base_dataset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                    train=True,
                                                    download=True)
        val_size = 500
    elif args.dataset == 'x_ray':
        base_dataset = X_Ray_Images(base_dir = args.data_dir,
                                    train = True,
                                    balanced = True,
                                    transform=transform_train)
        val_size = 50 # per class -> ± 12% of whole dataset


    # split dataset into labeled, unlabeled and validation sets
    train_labeled_idxs, \
    train_unlabeled_idxs, \
    val_idxs = train_val_split(labels = base_dataset.y,
                               n_labeled_per_class = int(args.n_labeled/args.num_classes),
                               number_classes = args.num_classes,
                               val_size = val_size)

    train_labeled_dataset = Labeled_Dataset(args = args,
                                            indexs = train_labeled_idxs,
                                            train = True,
                                            balanced = True,
                                            transform = transform_train)
    train_unlabeled_dataset = Unlabeled_Dataset(
                                            args = args,
                                            indexs = train_unlabeled_idxs,
                                            train = True,
                                            balanced = True,
                                            transform = TransformTwice(transform_train))

    val_dataset = Labeled_Dataset(args = args,
                                  indexs = val_idxs,
                                  train = True,
                                  transform = transform_val,
                                  download = True)

    test_dataset = Labeled_Dataset(args = args,
                                   train = False,
                                   transform = transform_val,
                                   download=True)

    print(f"#Labeled: {len(train_labeled_idxs)} ", end='')
    print("#Unlabeled: {len(train_unlabeled_idxs)} ", end='')
    print("#Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def get_data_loaders_no_ssl(args):

    # get datasets
    print(f'==> Preparing {args.dataset}')

    if args.dataset == 'cifar':
        crop_size = 32
        args.num_classes = 10
    elif args.dataset == 'x_ray':
        crop_size = 224
        args.num_classes = 7

    # transformations
    transform_train = transforms.Compose([
                        RandomPadandCrop(crop_size),
                        RandomFlip(),
                        ToTensor()])

    transform_val = transforms.Compose([
                        transforms.ToTensor()])

    train_set, test_set = get_datasets_no_ssl(
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     args = args)

    labeled_trainloader = DataLoader(train_set,
                                      batch_size = args.batch_size,
                                      shuffle = True,
                                      num_workers = 0,
                                      drop_last = True)
    test_loader = DataLoader(test_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=0)

    return labeled_trainloader, test_loader, args


def get_datasets_no_ssl(transform_train, transform_val, args):

    if args.dataset == 'cifar':
        train_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                    train=True,
                                                    download=True)
        test_set = torchvision.datasets.CIFAR10(root=args.data_dir,
                                                    train=False,
                                                    download=True)
    elif args.dataset == 'x_ray':
        train_set = Labeled_Dataset(base_dir = args.data_dir,
                                    train = True,
                                    balanced = True,
                                    transform=transform_train,
                                    args = args)
        test_set = Labeled_Dataset(base_dir = args.data_dir,
                                    train = False,
                                    balanced = True,
                                    transform=transform_train,
                                    args = args)
    return train_set, test_set


def train_val_split(labels, n_labeled_per_class, number_classes, val_size=500):

    if isinstance(labels, list):
        labels = np.array(labels)

    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(number_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-val_size])
        val_idxs.extend(idxs[-val_size:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def normalise(x, mean, std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

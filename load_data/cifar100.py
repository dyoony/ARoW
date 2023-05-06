import torch

import torchvision
import torchvision.transforms as transforms


def load_cifar100(data_dir, use_augmentation=True):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset
        
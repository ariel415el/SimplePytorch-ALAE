import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision import datasets as tv_datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split


def get_mnist(data_dir="data"):
    train_dataset = tv_datasets.MNIST(data_dir, train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    test_dataset = tv_datasets.MNIST(data_dir, train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))

    return train_dataset, test_dataset, 28

def get_celeba(data_dir="../data"):
    # train_dataset = tv_datasets.CelebA(data_dir, split='all', download=True)
    # test_dataset = tv_datasets.MNIST(data_dir, train=False, download=True)

    dataset = tv_datasets.ImageFolder(root=data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   # lambda x: x[0][None,:] # test with FC mode
                               ]))

    val_size = len(dataset) // 9
    train_dataset, val_dataset  = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset, 64


def get_dataset(dataset_name):
    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset, img_dim = get_mnist('../data')
    elif dataset_name.lower() == 'celeba':
        train_dataset, test_dataset, img_dim = get_celeba('../data')
    else:
        raise ValueError("No such available dataset")

    return train_dataset, test_dataset, img_dim


def get_dataloader(dataset, batch_size, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True}  # 'collate_fn': RequireGradCollator(device)
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


if __name__ == '__main__':
    data = get_celeba()
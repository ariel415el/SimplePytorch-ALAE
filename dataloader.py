import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision import datasets as tv_datasets
from torch.utils.data import Dataset
from sklearn import datasets as sk_datasets


def get_mnist(data_dir="data"):
    train_dataset = tv_datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = tv_datasets.MNIST(data_dir, train=False, download=True)

    train_data, train_labels = train_dataset.data.numpy().reshape(-1, 28*28), train_dataset.train_labels.numpy()
    test_data, test_labels = test_dataset.data.numpy().reshape(-1, 28*28), test_dataset.train_labels.numpy()

    return (train_data, train_labels, test_data, test_labels), "MNIST"


def get_sklearn_digits(plot=False):
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the 8x8 digits  data:
    digits = sk_datasets.load_digits()
    data = digits.data / 16 * 255.
    labels = digits.target

    d = int(data.shape[0]*0.9)
    train_data, train_labels = data[:d], labels[:d]
    test_data, test_labels = data[d:], labels[d:]
    return (train_data, train_labels, test_data, test_labels), "sklearn_digits"


class MNISTDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix / 127.5 - 1.

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        dim = int(np.sqrt(self.data_matrix.shape[1]))
        return self.data_matrix[idx].reshape(1, dim, dim)

    def get_data(self):
        return self.data_matrix


class RequireGradCollator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        with torch.no_grad():
            return torch.tensor(batch, requires_grad=True, device=self.device, dtype=torch.float32)


def get_dataloader(batch_size, device, test_set=False):
    data, _ = get_mnist('../data')
    # data, _ = get_sklearn_digits()
    data = data[2] if test_set else data[0]
    dataset = MNISTDataset(data)
    kwargs = {'batch_size': batch_size, 'collate_fn': RequireGradCollator(device)}
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                      )
    return torch.utils.data.DataLoader(dataset, **kwargs)


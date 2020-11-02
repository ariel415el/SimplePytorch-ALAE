import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision import datasets as tv_datasets
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mnist(data_dir="data"):
    train_dataset = tv_datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = tv_datasets.MNIST(data_dir, train=False, download=True)

    train_data, train_labels = train_dataset.data.numpy().reshape(-1, 28*28), train_dataset.train_labels.numpy()
    test_data, test_labels = test_dataset.data.numpy().reshape(-1, 28*28), test_dataset.train_labels.numpy()

    return (train_data, train_labels, test_data, test_labels), "MNIST"


def get_dataloader(batch_size):
    data, _ = get_mnist('../data')
    dataset = MNISTDataset(data[0].astype(np.float32))
    kwargs = {'batch_size': batch_size, 'collate_fn': RequireGradCollator()}
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                      )
    return torch.utils.data.DataLoader(dataset, **kwargs)


class MNISTDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix / 127.5 - 1.

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx].reshape(1,28,28)

    def get_data(self):
        return self.data_matrix


class RequireGradCollator(object):
    def __init__(self):
        self.device = device
    def __call__(self, batch):
        with torch.no_grad():
            return torch.tensor(batch, requires_grad=True, device=self.device, dtype=torch.float32)
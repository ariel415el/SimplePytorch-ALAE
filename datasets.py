import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision import datasets as tv_datasets
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.mnist import read_image_file
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
import os
import tarfile
import cv2

MNIST_WORKING_DIM=28
LFW_WORKING_DIM=64
VAL_SET_PORTION=0.05


def center_crop(img, size):
    y_start = int((img.shape[0] - size)/2)
    x_start = int((img.shape[1] - size)/2)
    return img[y_start: y_start + size, x_start: x_start + size]

def download_mnist(data_dir):
    """
    Taken from torchvision.datasets.mnist
    Dwonloads Mnist  from the official site
    reshapes themas images, normalizes them and saves them as a tensor
    """
    raw_folder = os.path.join(data_dir, 'raw')
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder, exist_ok=True)

        # download files
        train_imgs_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"
        test_imgs_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"
        for url, md5 in [train_imgs_url, test_imgs_url]:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=raw_folder, filename=filename, md5=md5)

    if not os.path.exists(os.path.join(data_dir, 'train_data.pt')):

        # process and save as torch files
        print('Processing...')

        training_set = read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte'))
        test_set = read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte'))

        # preprocess: reshape and normalize from [0,255] to [-1,1]
        training_set = training_set.reshape(-1, 1, MNIST_WORKING_DIM, MNIST_WORKING_DIM) / 127.5 - 1
        test_set = test_set.reshape(-1, 1, MNIST_WORKING_DIM, MNIST_WORKING_DIM) / 127.5 - 1

        with open(os.path.join(data_dir, 'train_data.pt'), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(data_dir, 'test_data.pt'), 'wb') as f:
            torch.save(test_set, f)

    print('Done!')


def download_lwf(data_dir):
    """
    Dwonloads LFW alligned images (deep funneled version) from the official site
    crops and normalizes them and saves them as a tensor
    """
    if not os.path.exists(os.path.join(data_dir, 'lfw-deepfunneled.tgz')):
        download_and_extract_archive("http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
                                     md5='68331da3eb755a505a502b5aacb3c201',
                                     download_root=data_dir, filename='lfw-deepfunneled.tgz')
    if not os.path.exists(os.path.join(data_dir, 'lfw-deepfunneled')):
        f = tarfile.open(os.path.join(data_dir, 'lfw-deepfunneled.tgz'), 'r:gz')
        f.extractall(data_dir)
        f.close()
    imgs = []
    if not os.path.exists(os.path.join(data_dir, 'all_imgs.pt')):
        for celeb_name in os.listdir(os.path.join(data_dir, 'lfw-deepfunneled')):
            for fname in os.listdir(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name)):
                img = cv2.imread(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name, fname))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = center_crop(img, 150)
                img = cv2.resize(img, (LFW_WORKING_DIM, LFW_WORKING_DIM))
                img = img.transpose(2,0,1)
                img = img / 127.5 - 1
                imgs.append(torch.tensor(img, dtype=torch.float32))
        with open(os.path.join(data_dir, 'all_imgs.pt'), 'wb') as f:
            torch.save(torch.stack(imgs), f)

    print('Done!')

def get_lfw(data_dir):
    """
    Returns an LFW train and val datalsets
    """
    download_lwf(data_dir)
    data = torch.load(os.path.join(data_dir, "all_imgs.pt"))

    data = data[:,0].reshape(-1, 1, 64, 64)

    dataset = SimpleDataset(data)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset, LFW_WORKING_DIM

def get_mnist(data_dir):
    """
    Returns an LFW train and val datalsets
    """
    download_mnist(data_dir)
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    train_dataset, val_dataset = SimpleDataset(train_data), SimpleDataset(test_data)

    return train_dataset, val_dataset, MNIST_WORKING_DIM


class SimpleDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx]

    def get_data(self):
        return self.data_matrix


def get_celeba(data_dir):
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

    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset, 64


def get_dataset(data_root, dataset_name):
    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset, img_dim = get_mnist(os.path.join(data_root, 'Mnist'))
    elif dataset_name.lower() == 'celeb-a':
        train_dataset, test_dataset, img_dim = get_celeba(os.path.join(data_root, 'Celeb-a'))
    elif dataset_name.lower() == 'lfw':
        train_dataset, test_dataset, img_dim = get_lfw(os.path.join(data_root, 'LFW'))

    else:
        raise ValueError("No such available dataset")

    return train_dataset, test_dataset, img_dim


class RequireGradCollator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        with torch.no_grad():
            # requires_grad=True is necessary for the gradient penalty calculation
            # return torch.tensor(batch, requires_grad=True, device=self.device, dtype=torch.float32)
            batch_tensor = torch.stack(batch).to(self.device).float()
            batch_tensor.requires_grad = True
            return batch_tensor


def get_dataloader(dataset, batch_size, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'collate_fn': RequireGradCollator(device)}
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


if __name__ == '__main__':
    get_mnist("../data/LFW")
    get_lfw("../data/LFW")
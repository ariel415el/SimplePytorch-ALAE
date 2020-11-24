import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive
from torchvision.datasets.mnist import read_image_file
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
import tarfile
import cv2
from tqdm import tqdm


MNIST_WORKING_DIM=28
VAL_SET_PORTION=0.05


class ImgLoader:
    def __init__(self, center_crop_size, resize, normalize, to_torch, dtype):
        self.center_crop_size = center_crop_size
        self.resize = resize
        self.normalize = normalize
        self.dtype = dtype
        self.to_torch = to_torch

    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.center_crop_size:
            img = center_crop(img, self.center_crop_size)
        if self.resize:
            img = cv2.resize(img, (self.resize, self.resize))
        img = img.transpose(2, 0, 1)
        if self.normalize:
            img = img / 127.5 - 1
        if self.to_torch:
            img = torch.tensor(img, dtype=self.dtype)
        else:
            img = img.astype(self.dtype)
        return img


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
        print("Downloadint LFW from official site...")
        download_and_extract_archive("http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
                                     md5='68331da3eb755a505a502b5aacb3c201',
                                     download_root=data_dir, filename='lfw-deepfunneled.tgz')
    if not os.path.exists(os.path.join(data_dir, 'lfw-deepfunneled')):
        f = tarfile.open(os.path.join(data_dir, 'lfw-deepfunneled.tgz'), 'r:gz')
        f.extractall(data_dir)
        f.close()


def download_celeba(data_dir):
    print("Downloading Celeb-a from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('jessicali9530/celeba-dataset', path=data_dir, unzip=True, quiet=False)
    print("Done!")

def download_ffhq_thumbnails(data_dir):
    print("Downloadint FFHQ-thumbnails from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('greatgamedota/ffhq-face-data-set', path=data_dir, unzip=True, quiet=False)
    print("Done.")


def get_lfw(data_dir, dim):
    """
    Returns an LFW train and val datalsets
    """
    download_lwf(data_dir)
    pt_name = f"LFW-{dim}x{dim}.pt"
    if not os.path.exists(os.path.join(data_dir, pt_name)):
        print("Preprocessing FFHQ data")
        imgs = []
        img_loader = ImgLoader(center_crop_size=150, resize=dim, normalize=True, to_torch=False, dtype=np.float32)
        for celeb_name in tqdm(os.listdir(os.path.join(data_dir, 'lfw-deepfunneled'))):
            for fname in os.listdir(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name)):
                img = img_loader(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name, fname))
                imgs.append(torch.tensor(img, dtype=torch.float32))
        with open(os.path.join(data_dir, pt_name), 'wb') as f:
            torch.save(torch.stack(imgs), f)

    data = torch.load(os.path.join(data_dir, pt_name))

    dataset = MemoryDataset(data)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset


def get_mnist(data_dir):
    """
    Returns an LFW train and val datalsets
    """
    download_mnist(data_dir)
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    train_dataset, val_dataset = MemoryDataset(train_data), MemoryDataset(test_data)

    return train_dataset, val_dataset, MNIST_WORKING_DIM


def get_celeba(data_dir, dim):
    imgs_dir = os.path.join(data_dir, 'img_align_celeba', 'img_align_celeba')
    if not os.path.exists(imgs_dir):
        download_celeba(data_dir)
    img_loader = ImgLoader(center_crop_size=170, resize=dim, normalize=True, to_torch=True, dtype=torch.float32)
    img_paths = [os.path.join(imgs_dir, fname) for fname in os.listdir(imgs_dir)]
    dataset = DiskDataset(img_paths, img_loader)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset

def get_ffhq(data_dir, dim):
    imgs_dir = os.path.join(data_dir, 'thumbnails128x128')
    if not os.path.exists(imgs_dir):
        download_ffhq_thumbnails(data_dir)

    pt_file = f"FFHQ_Thumbnail-{dim}x{dim}.pt"
    if not os.path.exists(os.path.join(data_dir, pt_file)):
        print("Preprocessing FFHQ data")
        imgs = []
        img_loader = ImgLoader(center_crop_size=None, resize=dim, normalize=True, to_torch=True, dtype=torch.float32)
        for img_name in tqdm(os.listdir(imgs_dir)):
            fname = os.path.join(imgs_dir, img_name)
            img = img_loader(fname)
            imgs.append(img)
        with open(os.path.join(data_dir, pt_file), 'wb') as f:
            torch.save(torch.stack(imgs), f)

    data = torch.load(os.path.join(data_dir, pt_file))
    dataset = MemoryDataset(data)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset

class MemoryDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx]

    def get_data(self):
        return self.data_matrix


class DiskDataset(Dataset):
    def __init__(self, image_paths, load_image_function):
        self.image_paths = image_paths
        self.load_image_function = load_image_function

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.load_image_function(self.image_paths[idx])


class EndlessDataloader:
    """
    An iterator wrapper for a dataloader that resets when reaches its end
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            real_image = next(self.iterator)

        except (OSError, StopIteration):
            self.iterator = iter(self.dataloader)
            real_image = next(self.iterator)

        return real_image


def get_dataset(data_root, dataset_name, dim):
    if dataset_name.lower() == 'mnist':
        assert dim == 28
        train_dataset, test_dataset, _ = get_mnist(os.path.join(data_root, 'Mnist'))
    elif dataset_name.lower() == 'celeb-a':
        train_dataset, test_dataset = get_celeba(os.path.join(data_root, 'Celeb-a'), dim)
    elif dataset_name.lower() == 'ffhq':
        train_dataset, test_dataset = get_ffhq(os.path.join(data_root, 'FFHQ-thumbnails'), dim)
    elif dataset_name.lower() == 'lfw':
        train_dataset, test_dataset = get_lfw(os.path.join(data_root, 'LFW'), dim)

    else:
        raise ValueError("No such available dataset")

    return train_dataset, test_dataset


class RequireGradCollator(object):
    def __init__(self, resize, device):
        self.device = device
        self.resize = resize

    def __call__(self, batch):
        with torch.no_grad():
            # requires_grad=True is necessary for the gradient penalty calculation
            # return torch.tensor(batch, requires_grad=True, device=self.device, dtype=torch.float32)
            batch_tensor = torch.stack(batch).to(self.device).float()
            if self.resize is not None:
                batch_tensor = torch.nn.functional.interpolate(batch_tensor, (self.resize, self.resize))
            batch_tensor.requires_grad = True
            return batch_tensor


def get_dataloader(dataset, batch_size, resize, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'collate_fn': RequireGradCollator(resize, device)}
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


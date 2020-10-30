import torch
from reproduce_ALAE.ALAE import Model
from reproduce_ALAE.dataloader import TFRecordsDataset, make_dataloader
from reproduce_ALAE.lod_driver import LODDriver
from reproduce_ALAE.custom_adam import LREQAdam
from reproduce_ALAE.tracker import LossTracker
from datasets import get_mnist
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image

BASE_LEARNING_RATE = 0.002
DECODER_LAYER_TO_RESOLUTION = 28
OUTPUT_DIR='training_outputs_DATASET'
EPOCHS=100
LATENT_SPACE_SIZE = 50
LAYER_COUNT = 4
MAPPING_LAYERS = 6
CHANNELS = 1  # Mnist is B&W
BATCH_SIZE=128

LOD_POWER = 5
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def get_dataloader(tf=True):
    data, _ = get_mnist('../data')
    dataset = SimpleDataset(data[0].astype(np.float32))
    kwargs = {'batch_size': BATCH_SIZE}
    if device != "cpu":
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )
    return torch.utils.data.DataLoader(dataset, **kwargs)



class SimpleDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx].reshape(1,28,28)

    def get_data(self):
        return self.data_matrix

def save_sample(epoch, tracker, sample, samplez, model):
    with torch.no_grad():
        model.eval()
        sample = sample[:BATCH_SIZE]
        samplez = samplez[:BATCH_SIZE]

        sample_in = sample
        while sample_in.shape[2] > DECODER_LAYER_TO_RESOLUTION:
            sample_in = F.avg_pool2d(sample_in, 2, 2)
        assert sample_in.shape[2] == DECODER_LAYER_TO_RESOLUTION

        Z, _ = model.encode(sample_in, LOD_POWER, 1)

        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

        rec1 = model.decoder(Z, LOD_POWER, 1, noise=False)
        rec2 = model.decoder(Z, LOD_POWER, 1, noise=True)


        Z = model.mapping_fl(samplez)
        g_rec = model.decoder(Z, LOD_POWER, 1, noise=True)
        sample_in = F.interpolate(sample_in, rec2.shape[2])

        resultsample = torch.cat([sample_in, rec1, rec2, g_rec], dim=0)

        def save_pic(x_rec):
            tracker.register_means(epoch)
            tracker.plot()

            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            f = os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % (epoch))
            save_image(result_sample, f, nrow=min(32, BATCH_SIZE))

        save_pic(resultsample)


def train_mnist():
    tracker = LossTracker(OUTPUT_DIR)

    model = Model(layer_count=LAYER_COUNT,
                  latent_size=LATENT_SPACE_SIZE,
                  mapping_layers=MAPPING_LAYERS,
                  channels=CHANNELS,
                  )
    model.train()

    test_model = Model(layer_count=LAYER_COUNT,
                  latent_size=LATENT_SPACE_SIZE,
                  mapping_layers=MAPPING_LAYERS,
                  channels=CHANNELS,
                  )
    test_model.eval()
    test_model.requires_grad_(False)

    decoder_optimizer = LREQAdam([
        {'params': model.decoder.parameters()},
        {'params': model.mapping_fl.parameters()}
    ], lr=BASE_LEARNING_RATE, betas=(0.0, 0.99), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': model.encoder.parameters()},
        {'params': model.mapping_tl.parameters()},
    ], lr=BASE_LEARNING_RATE, betas=(0.0, 0.99), weight_decay=0)


    # Create test dataset
    path = '/home/ariel/projects/ALAE/dataset_samples/mnist'
    src = []
    with torch.no_grad():
        for filename in list(os.listdir(path))[:32]:
            img = np.asarray(Image.open(os.path.join(path, filename)))[:, :, None]
            im = img.transpose((2, 0, 1))
            x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True) / 127.5 - 1.
            src.append(x)
        test_sample = torch.stack(src)

    test_samples_z = torch.tensor( np.random.RandomState(3456).randn(32, LATENT_SPACE_SIZE)).float()

    # dataloader = get_dataloader(tf=False)

    dataset = TFRecordsDataset(rank=0, world_size=1, buffer_size_mb=1024, channels=1)


    for epoch in range(EPOCHS):
        i=0

        dataset.reset(LOD_POWER, BATCH_SIZE)
        dataloader = make_dataloader(dataset, BATCH_SIZE)

        for x_orig in tqdm(dataloader):
            i += 1
            with torch.no_grad():
                if x_orig.shape[0] != BATCH_SIZE:
                    print("Skipping partial batch")
                    continue
                x_orig = (x_orig / 127.5 - 1.)
                x = x_orig

            x.requires_grad = True

            encoder_optimizer.zero_grad()
            loss_d = model(x, LOD_POWER, 1, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(x, LOD_POWER, 1, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, LOD_POWER, 1, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            betta = 0.5 ** (BATCH_SIZE / (10 * 1000.0))
            test_model.lerp(model, betta)

        save_sample(epoch, tracker, test_sample, test_samples_z, test_model)


if __name__ == '__main__':
    train_mnist()
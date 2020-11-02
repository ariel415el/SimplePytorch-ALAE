import torch
from ALAE import Model
from custom_adam import LREQAdam
from tracker import LossTracker
from dataloader import get_dataloader
from tqdm import tqdm
import os
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image

BASE_LEARNING_RATE = 0.002
DECODER_LAYER_TO_RESOLUTION = 28
OUTPUT_DIR= 'training_outputs_DATASET_old'
EPOCHS=1000
LATENT_SPACE_SIZE = 50
LAYER_COUNT = 4
MAPPING_LAYERS = 6
CHANNELS = 1  # Mnist is B&W
BATCH_SIZE=128

LOD_POWER = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def save_sample(epoch, tracker, sample, samplez, model):
    with torch.no_grad():
        model.eval()

        Z, _ = model.encode(sample)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        sample_recreation = model.decoder(Z, LOD_POWER)

        Z = model.mapping_fl(samplez)
        generation_from_random = model.decoder(Z, LOD_POWER)
        sample_upscaled = F.interpolate(sample, sample_recreation.shape[2])

        resultsample = torch.cat([sample_upscaled, sample_recreation, generation_from_random], dim=0)

        resultsample = (resultsample*0.5 + 0.5).cpu()

        tracker.register_means(epoch)
        tracker.plot()
        f = os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % (epoch))
        save_image(resultsample, f, nrow=min(32, BATCH_SIZE))


def train_mnist():
    tracker = LossTracker(OUTPUT_DIR)

    model = Model(layer_count=LAYER_COUNT,
                  latent_size=LATENT_SPACE_SIZE,
                  mapping_layers=MAPPING_LAYERS,
                  channels=CHANNELS, device=device
                  )
    model.train().to(device)
    test_model = Model(layer_count=LAYER_COUNT,
                  latent_size=LATENT_SPACE_SIZE,
                  mapping_layers=MAPPING_LAYERS,
                  channels=CHANNELS, device=device
                  )
    test_model.eval().to(device)
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

    dataloader = get_dataloader(BATCH_SIZE, device=device)
    test_samples_z = torch.tensor(np.random.RandomState(3456).randn(32, LATENT_SPACE_SIZE), dtype=torch.float32).to(device)
    test_dataloader = get_dataloader(batch_size=32, device=device, test_set=True)
    test_samples = next(iter(test_dataloader))

    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader):
            encoder_optimizer.zero_grad()
            loss_d = model(batch, LOD_POWER, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(batch, LOD_POWER, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(batch, LOD_POWER, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            betta = 0.5 ** (BATCH_SIZE / (10 * 1000.0))
            test_model.lerp(model, betta)

        save_sample(epoch, tracker, test_samples, test_samples_z, test_model)


if __name__ == '__main__':
    train_mnist()
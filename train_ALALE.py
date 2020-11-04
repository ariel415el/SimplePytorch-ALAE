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

OUTPUT_DIR= 'training_outputs_DATASET'
BASE_LEARNING_RATE = 0.002
EPOCHS=1000
LATENT_SPACE_SIZE = 50
MAPPING_LAYERS = 6
BATCH_SIZE=128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_sample(epoch, tracker, sample, samplez, model):
    """

    Save a debug image containig real images, their reconstruction and fake generated images
    """
    with torch.no_grad():
        model.eval()

        W, _ = model.encode(sample)
        sample_recreation = model.G(W)

        W = model.F(samplez)
        generation_from_random = model.G(W)

        tracker.update(dict(gen_min_val=0.5*(sample_recreation.min() + generation_from_random.min()),
                            gen_max_val=0.5*(sample_recreation.max() + generation_from_random.max())))


        resultsample = torch.cat([sample, sample_recreation, generation_from_random], dim=0).cpu()

        # Normalize images from -1,1 to 0, 1.
        # Eventhough train samples are in this range (-1,1), the generated image may not. But this should diminish as
        # raining continues or else the discriminator can detect them. Anyway save_image clamps it to 0,1
        resultsample = resultsample * 0.5 + 0.5

        tracker.register_means(epoch)
        tracker.plot()
        f = os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % (epoch))
        save_image(resultsample, f, nrow=min(32, BATCH_SIZE))


def train_mnist():
    tracker = LossTracker(OUTPUT_DIR)

    model = Model(latent_size=LATENT_SPACE_SIZE, mapping_layers=MAPPING_LAYERS,  device=device)
    model.train().to(device)

    decoder_optimizer = LREQAdam([
        {'params': model.G.parameters()},
        {'params': model.F.parameters()}
    ], lr=BASE_LEARNING_RATE, betas=(0.0, 0.99), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': model.E.parameters()},
        {'params': model.D.parameters()},
    ], lr=BASE_LEARNING_RATE, betas=(0.0, 0.99), weight_decay=0)

    # Create test dataset

    dataloader = get_dataloader(BATCH_SIZE, device=device)
    test_samples_z = torch.tensor(np.random.RandomState(3456).randn(32, LATENT_SPACE_SIZE), dtype=torch.float32).to(device)
    test_dataloader = get_dataloader(batch_size=32, device=device, test_set=True)
    test_samples = next(iter(test_dataloader))

    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader):
            encoder_optimizer.zero_grad()
            loss_d = model(batch, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(batch, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(batch, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        save_sample(epoch, tracker, test_samples, test_samples_z, model)


if __name__ == '__main__':
    train_mnist()
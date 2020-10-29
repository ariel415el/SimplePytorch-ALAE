import torch
from reproduce_ALAE.ALAE import Model
from reproduce_ALAE.dataloader import TFRecordsDataset, make_dataloader
from reproduce_ALAE.lod_driver import LODDriver
from reproduce_ALAE.custom_adam import LREQAdam
from reproduce_ALAE.tracker import LossTracker
from tqdm import tqdm
import os
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image

BASE_LEARNING_RATE = 0.0001
DECODER_LAYER_TO_RESOLUTION = 28
OUTPUT_DIR='training_outputs'
EPOCHS=100
LATENT_SPACE_SIZE = 50
LAYER_COUNT = 4
MAPPING_LAYERS = 6
CHANNELS = 1  # Mnist is B&W

def save_sample(lod2batch, tracker, sample, samplez, model):
    with torch.no_grad():
        model.eval()
        sample = sample[:lod2batch.get_batch_size()]
        samplez = samplez[:lod2batch.get_batch_size()]

        sample_in = sample
        while sample_in.shape[2] > DECODER_LAYER_TO_RESOLUTION:
            sample_in = F.avg_pool2d(sample_in, 2, 2)
        assert sample_in.shape[2] == DECODER_LAYER_TO_RESOLUTION

        Z, _ = model.encode(sample_in, lod2batch.lod, 1)

        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

        rec1 = model.decoder(Z, lod2batch.lod, 1, noise=False)
        rec2 = model.decoder(Z, lod2batch.lod, 1, noise=True)


        Z = model.mapping_fl(samplez)
        g_rec = model.decoder(Z, lod2batch.lod, 1, noise=True)
        sample_in = F.interpolate(sample_in, rec2.shape[2])

        resultsample = torch.cat([sample_in, rec1, rec2, g_rec], dim=0)

        def save_pic(x_rec):
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            f = os.path.join(OUTPUT_DIR,
                             'sample_%d_%d.jpg' % (
                                 lod2batch.current_epoch + 1,
                                 lod2batch.iteration // 1000)
                             )
            save_image(result_sample, f, nrow=min(32, lod2batch.get_batch_size()))

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

    dataset = TFRecordsDataset(rank=0, world_size=1, buffer_size_mb=1024, channels=1)

    lod2batch = LODDriver(dataset_size=len(dataset))

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

    for epoch in range(EPOCHS):
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_batch_size())
        batches = make_dataloader(dataset, lod2batch.get_batch_size())

        i=0
        for x_orig in tqdm(batches):
            i += 1
            with torch.no_grad():
                if x_orig.shape[0] != lod2batch.get_batch_size():
                    print("Skipping partial batch")
                    continue
                x_orig = (x_orig / 127.5 - 1.)
                x = x_orig

            x.requires_grad = True

            encoder_optimizer.zero_grad()
            loss_d = model(x, lod2batch.lod, 1, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(x, lod2batch.lod, 1, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, lod2batch.lod, 1, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
            test_model.lerp(model, betta)

            lod2batch.step()


        save_sample(lod2batch, tracker, test_sample, test_samples_z, test_model)

if __name__ == '__main__':
    train_mnist()
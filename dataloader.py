# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dareblopy as db
import random

import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
import time
import math

cpu = torch.device('cpu')


class TFRecordsDataset:
    def __init__(self, rank=0, world_size=1, buffer_size_mb=200, channels=3, seed=None, train=True, needs_labels=False):
        self.rank = rank
        self.last_data = ""
        if train:
            self.part_count = 1
            self.part_size = 60000 // self.part_count
        else:
            self.part_count = 1
            self.part_size = 10000 // self.part_count
        self.workers = []
        self.workers_active = 0
        self.iterator = None
        self.filenames = {}
        self.batch_size = 512
        self.features = {}
        self.channels = channels
        self.seed = seed
        self.train = train
        self.needs_labels = needs_labels

        assert self.part_count % world_size == 0

        self.part_count_local = self.part_count // world_size

        if train:
            path = '/data/datasets/mnist32/tfrecords/mnist-r%02d.tfrecords.%03d'
        else:
            path = '/data/datasets/mnist32/tfrecords/mnist_test-r%02d.tfrecords.%03d'

        for r in range(2, 5 + 1):
            files = []
            for i in range(self.part_count_local * rank, self.part_count_local * (rank + 1)):
                file = path % (r, i)
                files.append(file)
            self.filenames[r] = files

        self.buffer_size_b = 1024 ** 2 * buffer_size_mb

        self.current_filenames = []

    def reset(self, lod, batch_size):
        assert lod in self.filenames.keys()
        self.current_filenames = self.filenames[lod]
        self.batch_size = batch_size

        img_size = 2 ** lod

        if self.needs_labels:
            self.features = {
                # 'shape': db.FixedLenFeature([3], db.int64),
                'data': db.FixedLenFeature([self.channels, img_size, img_size], db.uint8),
                'label': db.FixedLenFeature([], db.int64)
            }
        else:
            self.features = {
                # 'shape': db.FixedLenFeature([3], db.int64),
                'data': db.FixedLenFeature([self.channels, img_size, img_size], db.uint8)
            }

        buffer_size = self.buffer_size_b // (self.channels * img_size * img_size)

        if self.seed is None:
            seed = np.uint64(time.time() * 1000)
        else:
            seed = self.seed

        self.iterator = db.ParsedTFRecordsDatasetIterator(self.current_filenames, self.features, self.batch_size, buffer_size, seed=seed)

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.part_count_local * self.part_size


def make_dataloader(dataset, GPU_batch_size):
    class BatchCollator(object):
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, batch):
            with torch.no_grad():
                x, = batch
                x = torch.tensor(x, requires_grad=True, device=torch.device(self.device), dtype=torch.float32)
                return x

    batches = db.data_loader(iter(dataset), BatchCollator(), len(dataset) // GPU_batch_size)

    return batches
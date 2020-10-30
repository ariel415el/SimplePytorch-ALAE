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

import torch
import math
import time
from collections import defaultdict
torch.manual_seed(7)
MODEL_LAYER_COUNT = 4
SNAPSHOT_FREQ = [300, 300, 300, 300, 300]
REPORT_FREQ = [100, 300, 300, 300, 300]


class LODDriver:
    def __init__(self, dataset_size):

        self.lod_2_batch = [128, 128, 128, 128]

        self.minibatch_base = 16
        self.dataset_size = dataset_size
        self.current_epoch = 0
        self.lod = -1
        self.iteration = 0
        self.epoch_end_time = 0
        self.epoch_start_time = 0
        self.per_epoch_ptime = 0
        self.reports = REPORT_FREQ
        self.snapshots = SNAPSHOT_FREQ
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0

    def get_lod_power2(self):
        return self.lod + 2

    def get_batch_size(self):
        return self.lod_2_batch[min(self.lod, len(self.lod_2_batch) - 1)]

    def get_dataset_size(self):
        return self.dataset_size

    def is_time_to_report(self):
        if self.iteration >= self.tick_start_nimg_report + self.reports[min(self.lod, len(self.reports) - 1)] * 1000:
            self.tick_start_nimg_report = self.iteration
            return True
        return False

    def is_time_to_save(self):
        if self.iteration >= self.tick_start_nimg_snapshot + self.snapshots[min(self.lod, len(self.snapshots) - 1)] * 1000:
            self.tick_start_nimg_snapshot = self.iteration
            return True
        return False

    def step(self):
        self.iteration += self.get_batch_size()
        self.epoch_end_time = time.time()
        self.per_epoch_ptime = self.epoch_end_time - self.epoch_start_time

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.iteration = 0
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0
        self.epoch_start_time = time.time()

        self.lod = MODEL_LAYER_COUNT - 1

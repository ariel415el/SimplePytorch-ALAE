from collections import defaultdict
import matplotlib.pyplot as plt
import os


class RunningMean:
    def __init__(self):
        self.mean = 0.0
        self.n = 0
        self.means = []

    def add(self, value):
        self.mean = (float(value) + self.mean * self.n)/(self.n + 1)
        self.n += 1
        return self

    def reset(self):
        self.mean = 0.0
        self.n = 0

    def get_means(self):
        self.means += [self.mean]
        self.reset()
        return self.means


class LossTracker:
    def __init__(self, output_folder):
        self.tracks = defaultdict(RunningMean)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def update(self, d):
        for k, v in d.items():
            self.tracks[k].add(v.item())

    def plot(self):
        plt.figure(figsize=(12, 8))
        for key in self.tracks.keys():
            plot = self.tracks[key].get_means()
            plt.plot(range(len(plot)), plot, label=key)

        plt.xlabel('steps')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_folder, 'plot.png'))
        plt.close()

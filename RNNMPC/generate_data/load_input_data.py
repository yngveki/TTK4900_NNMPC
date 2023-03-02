#!/usr/bin/env python3

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# TODO:
# 1) Dataset should be able to receive multiple paths (maybe by means of *args and an assertion that they are all strings?),
#    such that many different input profiles can be merged into the same training session
class SRDataset(Dataset):
    """SR == Step Response"""
    def __init__(self, csv_path, mu=1, my=1):
        """
        Args
            :param csv_path: path to data location
            :param mu: amount of historical actuation data per sample. This excludes current!
            :param my: amount of historical output data per sample. This excludes current!
        """

        df = pd.read_csv(csv_path)

        self.u1_full = np.asarray(df.iloc[0,:])
        self.u2_full = np.asarray(df.iloc[1,:])
        self.y1_full = np.asarray(df.iloc[2,:])
        self.y2_full = np.asarray(df.iloc[3,:])

        if all((len(self.u1_full) != len(self.u2_full), 
                len(self.u1_full) != len(self.y1_full), 
                len(self.u1_full) != len(self.y2_full))):
            Exception('All data sequences must be equally long!\n')
        self.data_length = len(self.u1_full)

        self.mu = mu
        self.my = my
        largest_m = np.max((mu, my))
        num_samples = self.data_length - largest_m - 1 # We want the last sample to also have a next, for label's sake
        sr = []
        for n in range(num_samples):
            # .copy() because tensor's can't handle negative stride. Could alternately flip *_full, '
            # flip iteration and alter indexing, but that seems worse

            if n == 0: n = -self.data_length # So that the first loop will also grab first element in arrays (because n=0 - 1 otherwise becomes the last index)
            u1 = self.u1_full[n + mu:n - 1:-1].copy() # current->past
            u2 = self.u2_full[n + mu:n - 1:-1].copy() # current->past
            y1 = self.y1_full[n + my:n - 1:-1].copy() # current->past
            y2 = self.y2_full[n + my:n - 1:-1].copy() # current->past
            k = [n + t for t in range(largest_m)]
            sample = {'u1': u1, 
                      'u2': u2, 
                      'y1': y1, 
                      'y2': y2, 
                      'k': k, 
                      'mu': mu, 
                      'my': my, 
                      'target': [self.y1_full[n + my + 1], self.y2_full[n + my + 1]]}
            sr.append(sample)

        self.SR = sr

    def __len__(self):
        """Returns number of samples available"""
        return len(self.SR)

    def __getitem__(self, index):
        """Gets a sample from the dataset"""
        return self.SR[index]

    def __repr__(self):
        "Represents the class more pretty in debugger"
        return f"Step Response dataset, length: {len(self)}"

    def plot_sr(self, idx=0, rand=False, delta_t=10):
        """Shows step response nr. idx. Random step response if rand=True"""
        if rand:
            idx = np.random.randint(0, self.__len__() - 1)
        
        sample = self[idx]
        y1 = sample['y1'][:]
        y2 = sample['y2'][:]
        u1 = sample['u1'][:]
        u2 = sample['u2'][:]
        t = np.multiply(sample['k'][:], delta_t)

        fig, axs = plt.subplots(2, 2)
        t = np.linspace(0, len(self) * delta_t, num=len(self))

        axs[0,0].plot(t[:self.my], y1, label='gas rate', color='tab:orange', marker='o')
        axs[0,0].legend()
        axs[1,0].plot(t[:self.mu], u1, label='choke', marker='o')
        axs[1,0].legend()
        axs[0,1].plot(t[:self.my], y2, label='oil rate', color='tab:orange', marker='o')
        axs[0,1].legend()
        axs[1,1].plot(t[:self.mu], u2, label='GL rate', marker='o')
        axs[1,1].legend()
        
        fig.suptitle('Step response')
        fig.tight_layout()

        plt.show()

    def plot_full(self, delta_t=10):

        fig, axs = plt.subplots(2, 2)
        t = np.linspace(0, self.data_length * delta_t, num=self.data_length)
        axs[0,0].plot(t, self.y1_full, '-', label='gas rate', color='tab:orange')
        axs[0,0].legend()
        axs[1,0].plot(t, self.u1_full, '-', label='choke')
        axs[1,0].legend()
        axs[0,1].plot(t, self.y2_full, '-', label='oil rate', color='tab:orange')
        axs[0,1].legend()
        axs[1,1].plot(t, self.u2_full, '-', label='GL rate')
        axs[1,1].legend()

        fig.suptitle('Step response')
        fig.tight_layout()

        plt.show()

# ----- Actual function ----- #
def load_input_data(csv_path, bsz=64, mu=1, my=1, shuffle=False):
    
    sr_dataset = SRDataset(csv_path, mu=mu, my=my)

    return DataLoader(sr_dataset, batch_size=bsz, shuffle=shuffle, num_workers=0)


# ----- For testing ----- #
if __name__ == "__main__":

    from pathlib import Path
    csv_path = Path(__file__).parent / "data/step_choke_0_2_2000.csv"

    sr = SRDataset(csv_path=csv_path)
    sr.plot_full()
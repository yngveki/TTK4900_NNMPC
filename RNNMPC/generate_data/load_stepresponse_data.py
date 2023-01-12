# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch.utils.data.sampler import SubsetRandomSampler

from pathlib import Path
from yaml import safe_load

# Ignore warnings
import warnings

class SRDataset(Dataset):
    """SR == Step Response"""
    def __init__(self, csv_path, config_path, mu=1, my=1):

        sr_frame = pd.read_csv(csv_path)
        u1_full = np.asarray(sr_frame.iloc[0,:])
        u2_full = np.asarray(sr_frame.iloc[1,:])
        y1_full = np.asarray(sr_frame.iloc[2,:])
        y2_full = np.asarray(sr_frame.iloc[3,:])

        if all((len(u1_full) != len(u2_full), 
                len(u1_full) != len(y1_full), 
                len(u1_full) != len(y2_full))):
            Exception('All elements of initial data must have equally many components.\n')
        data_length = len(u1_full)

        self.mu = mu
        self.my = my
        largest_m = np.max((mu, my))
        num_samples = data_length - largest_m - 1 # We want the last sample to also have a next, for label's sake
        sr = []
        for n in range(num_samples):
            u1 = u1_full[n:n + mu]
            u2 = u2_full[n:n + mu]
            y1 = y1_full[n:n + my]
            y2 = y2_full[n:n + my]
            k = [n + t for t in range(largest_m)]
            sample = {'u1': u1, 
                      'u2': u2, 
                      'y1': y1, 
                      'y2': y2, 
                      'k': k, 
                      'mu': mu, 
                      'my': my, 
                      'target': [y1_full[n + my], y2_full[n + my]]}
            sr.append(sample)

        self.SR = sr

        # TODO: this bit is an artifact
        with open(config_path, "r") as f:
            config = safe_load(f)
            if not config:
                Exception("Failed loading config file\n")
                
        self.start_time = config['start_time']
        self.warm_start_t = config['warm_start_t']
        self.step_time = self.warm_start_t + config['step_time'] # At what time the step should occur on the input
        self.final_time = self.warm_start_t + config['final_time'] # For how long the total simulation will last
        self.delta_t = config['delta_t']


    def split_train_val(self, ratio_train=0.8, shuffle=False):
        assert ratio_train >= 0 and ratio_train <= 1, "Probability must be within (0,1)"

        if ratio_train == 1:
            return self.SR, None

        dataset_size = len(self)
        indices = list(range(dataset_size))
        train_val_split = int(np.floor(dataset_size * ratio_train))

        if shuffle:
            # np.random.seed(1)
            np.random.shuffle(indices)
        
        train_indices = indices[:train_val_split]
        val_indices = indices[train_val_split:]

        train_loader = SubsetRandomSampler(train_indices)
        val_loader = SubsetRandomSampler(val_indices)

        return train_loader, val_loader

    def __len__(self):
        """Returns number of samples available"""
        return len(self.SR)

    def __getitem__(self, index):
        """Gets a sample from the dataset"""
        return self.SR[index]

    def __repr__(self):
        "Represents the class more pretty in debugger"
        return "Step Response dataset"

    def show_sr(self, idx=0, rand=False):
        """Shows step response nr. idx. Random step response if rand=True"""
        if rand:
            idx = np.random.randint(0, self.__len__() - 1)
        
        sample = self[idx]
        y1 = sample['y1'][:]
        y2 = sample['y2'][:]
        u1 = sample['u1'][:]
        u2 = sample['u2'][:]
        t = np.multiply(sample['k'][:], self.delta_t)

        fig, axs = plt.subplots(2, 2)
        t = np.linspace(0, self.__len__() * self.delta_t, num=self.__len__())

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


# ----- Actual function ----- #
def load_stepresponse_data(csv_path, config_path, train=True, bsz=64, mu=1, my=1, shuffle=False):

    warnings.filterwarnings("ignore")

    config_path = Path(__file__).parent / "../config/generate_data.yaml"
    sr_dataset = SRDataset(csv_path, config_path, mu=mu, my=my)

    return DataLoader(sr_dataset, batch_size=bsz, shuffle=shuffle, num_workers=0)

    # if train:    
    #     train_ratio = 0.8 
    #     sr_train_loader, sr_val_loader = sr_dataset.split_train_val(train_ratio, shuffle)

    #     return DataLoader(sr_dataset, batch_size=bsz, num_workers=0, sampler=sr_train_loader), \
    #            DataLoader(sr_dataset, batch_size=bsz, num_workers=0, sampler=sr_val_loader)

    # else:
    #     return DataLoader(sr_dataset, batch_size=bsz, shuffle=shuffle, num_workers=0)

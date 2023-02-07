#!/usr/bin/env python3

# Implements a class that holds the neural network

# Ref: https://github.com/bgrimstad/TTK28-Courseware/blob/master/model/flow_model.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from os.path import exists
from yaml import safe_load


# ----- NEURAL NETWORK ----- #
class NeuralNetwork(nn.Module):
    
    def __init__(self, layers=None, model_path=None):
        """
        Initializes a network with the given structural hyperparameters.

        Args:
            :param layers: The sizes of the the input, hidden and output layers to be defined.
        """
        super(NeuralNetwork, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert layers != None, "Your need to provide sizes for the layers!"
        assert len(layers) >= 3, "You need to define at least an input, hidden and output layer"
        self.layers = layers

        temp = []
        for idx in range(len(layers) - 1):
            temp.append(nn.Linear(layers[idx], layers[idx + 1]))

        self.layers = nn.ModuleList(temp)

        self.act = torch.nn.ReLU(inplace=False)

        if model_path is not None:
            try:
                self.load(model_path)
            except ValueError:
                print(f"Given path {model_path} doesn\'t hold a model!")

    def forward(self, x):
        """
        Performs a forward pass of the input data x and returns the output.
        
        Args:
            :param x: the given input data to be fed through the neural network.
        """
        input = x
        for l in self.layers[:-1]:
            input = self.act(l(input))

        return self.layers[-1](input)

    def log_MSE(self, mse):
        """
        Logs the mean square error for either validation or testing.
        
        Args:
            :param mse: the mean square error value to be logged.
        """
        self.mse = mse

    def save(self, path):
        """
        Uncritically saves model to the given path. Whether a model already exists
        at given path has to be done at higher level.

        Args:
            :param path: the given path at which to save self.state_dict.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads the model at the given path into the current instance of self.

        Args:
            :param path: the given path from which to load the model.
        """
        self.load_state_dict(torch.load(path))

        # Default to evaluation mode when loaded
        self.eval()

    def extract_coefficients(self):
        """
        Returns weights and biases for the current model as dicts
        """

        weights = {}
        biases = {}

        return NotImplementedError
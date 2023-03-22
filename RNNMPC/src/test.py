#!/usr/bin/env python3

# Tests a neural network for SingleWell model estimation on some given dataset

import torch
import torch.nn as nn

from src.neuralnetwork import NeuralNetwork
from generate_data.load_input_data import load_input_data

# ----- PREDICTING A SEQUENCE ----- #
def test(model_path, csv_path, hyperparameters):
    
    mu = hyperparameters['mu']
    my = hyperparameters['my']
    hlszs = hyperparameters['hlszs']
    batch_size = hyperparameters['bsz']

    layers = []
    layers.append(2 * (mu + 1) + 2 * (my + 1)) # Input layer
    layers += hlszs                            # Hidden layers
    layers.append(2)                           # Output layer
    model = NeuralNetwork(layers=layers, model_path=model_path)
    loss_fn = nn.MSELoss() # MSELoss since we're regressing

    # TODO: Change into smaller subset of huge dataset, such that potential bias in data-creation (se spørsmål til Lars, 2023.03.20) is not an issue
    test_dl = load_input_data(csv_path, bsz=batch_size, mu=mu, my=my)

    predicted = {'y1': [],
                 'y2': [],
                 'bias y1': [],
                 'bias y2': []}
    with torch.no_grad():
        for sample in test_dl.dataset:
            u1 = torch.Tensor(sample['u1'])
            u2 = torch.Tensor(sample['u2'])
            y1 = torch.Tensor(sample['y1'])
            y2 = torch.Tensor(sample['y2'])
            truth = torch.Tensor(sample['target'])
            X = torch.cat((u1, u2, y1, y2))

            # Compute prediction and loss (loss is our metric for accuracy)
            pred = model(X) 
            predicted['y1'].append(pred[0])
            predicted['y2'].append(pred[1])

            test_loss = loss_fn(pred, truth).item()

            # Log bias
            predicted['bias y1'].append(truth[0] - pred[0])
            predicted['bias y2'].append(truth[1] - pred[1])

    
    num_batches = len(test_dl)
    test_loss /= num_batches
    print(f"Avg (normalized) loss: {test_loss:>8f} \n")

    model.log_MSE(test_loss)

    return predicted, model
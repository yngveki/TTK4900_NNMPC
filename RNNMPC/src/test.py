#!/usr/bin/env python3

# Tests a neural network for SingleWell model estimation on some given dataset

import torch
import torch.nn as nn

from generate_data.load_input_data import load_input_data

# ----- PREDICTING A SEQUENCE ----- #
def test(model, csv_path, hyperparameters):
    
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    batch_size = hyperparameters['TRAINING']['bsz']

    loss_fn = nn.MSELoss() # MSELoss since we're regressing

    test_dl = load_input_data(csv_path, bsz=batch_size, mu=mu, my=my)

    predicted = {}
    predicted['y1'] = []
    predicted['y2'] = []
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

    
    num_batches = len(test_dl)
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

    model.log_MSE(test_loss)

    return predicted
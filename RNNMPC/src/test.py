#!/usr/bin/env python3

# Tests a neural network for SingleWell model estimation on some given dataset

import torch
import torch.nn as nn

from src.neuralnetwork import NeuralNetwork
from generate_data.load_input_data import load_input_data

#! TEMP
from yaml import safe_load
from pathlib import Path

# ----- PREDICTING A SEQUENCE ----- #
def test(model_path, csv_path, hyperparameters):
    
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    hlszs = hyperparameters['STRUCTURE']['hlszs']
    batch_size = hyperparameters['LEARNING']['bsz']

    layers = []
    layers.append(2 * (mu + 1) + 2 * (my + 1)) # Input layer
    layers += hlszs                            # Hidden layers
    layers.append(2)                           # Output layer
    model = NeuralNetwork(layers=layers, model_path=model_path)
    loss_fn = nn.MSELoss() # MSELoss since we're regressing

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
    print(f"Avg loss: {test_loss:>8f} \n")

    model.log_MSE(test_loss)

    return predicted, model

#! TEMP FOR DEBUGGING OF RNN IN RNNMPC
def test_single_sample(model_path, hyperparameters, input_vector):
    
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    hlszs = hyperparameters['STRUCTURE']['hlszs']

    layers = []
    layers.append(2 * (mu + 1) + 2 * (my + 1)) # Input layer
    layers += hlszs                            # Hidden layers
    layers.append(2)                           # Output layer
    model = NeuralNetwork(layers=layers, model_path=model_path)

    return model(input_vector)

if __name__ == '__main__':
    model_path = Path(__file__).parent / 'models/model_mixed_ramp_10/model_mixed_ramp_10.pt'
    hyperparameter_path = Path(__file__).parent / 'models/model_mixed_ramp_10/model_mixed_ramp_10.yaml'
    with open(hyperparameter_path, "r") as f:
        hyperparameters = safe_load(f)
    x = [51.09997683084354,
         50.54999885482869,
         50,
         50,
         50,
         50,
         333.400003011535,
         166.7000015077866,
         0,
         0,
         0,
         0,
         6417.297146506286,
         6417.297146506286,
         6407.312985605277,
         6407.312985605277,
         6407.312985605277,
         6407.312985605277,
         245.34721065469768,
         245.34721065469768,
         244.93704375989967,
         244.34587277953147,
         244.9549418168157,
         244.9642564604021]
    ret = test_single_sample(model_path, hyperparameters, input_vector=x)

    RNNMPCs_result = [2156.48, 1185.6]
    for idx, el in enumerate(ret):
        print(f'diff: {el - RNNMPCs_result[idx]}')
    print(ret)
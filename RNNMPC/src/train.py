#!/usr/bin/env python3

# Trains neural network for SingleWell model estimation on some given dataset

import torch
import torch.nn as nn
from pathlib import Path

from src.neuralnetwork import NeuralNetwork
from generate_data.load_stepresponse_data import load_stepresponse_data

class EarlyStopping():

    def __init__(self, p=5, min_delta=0, restore_best_weights=True):
        self.patience = p
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = model

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        return False

# ----- UTIL FUNCTIONS ----- #
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    total_itrs = 0
    tot_loss = 0
    for batch, sample in enumerate(dataloader.dataset):
        # Ensure zero-state from any previous iteration
        optimizer.zero_grad()

        # Prepare format of data
        u1 = torch.Tensor(sample['u1'][:model.mu])
        u2 = torch.Tensor(sample['u2'][:model.mu])
        y1 = torch.Tensor(sample['y1'][:model.my])
        y2 = torch.Tensor(sample['y2'][:model.my])
        X = torch.cat((u1, u2, y1, y2)) # Concatenating lists for single input-vector
        truth = torch.Tensor(sample['target'])

        # Compute prediction and loss
        pred = model(X) 
        loss = loss_fn(pred, truth)

        tot_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        total_itrs += 1
    
    return (tot_loss / total_itrs).item()

def validation_loop(dataloader, model, epoch):
    model.eval()
    mse_val = 0
    for sample in dataloader.dataset:
        # Prepare format of data
        # TODO: Is indexing superfluous? is not e.g. sample['u1'] sufficient?
        u1 = torch.Tensor(sample['u1'][:model.mu])
        u2 = torch.Tensor(sample['u2'][:model.mu])
        y1 = torch.Tensor(sample['y1'][:model.my])
        y2 = torch.Tensor(sample['y2'][:model.my])
        X = torch.cat((u1, u2, y1, y2))
        truth = torch.Tensor(sample['target'])

        mse_val += torch.sum(torch.pow(truth - model(X), 2)).item()
    
    mse_val /= len(dataloader.dataset)
    print(f'Epoch: {epoch + 1}: Validation MSE: {mse_val}')
    return mse_val

# ----- TRAINING ----- #
def train_NN(hyperparameters, csv_path_train, csv_path_val):

    #! TEMP
    gen_data_config_path = Path(__file__).parent / "../config/generate_data.yaml"
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    batch_size = hyperparameters['TRAINING']['bsz']
    train_dl = load_stepresponse_data(csv_path_train, gen_data_config_path, train=True,
                                            bsz=batch_size, mu=mu, my=my, shuffle=True)
                                            
    n_MV = hyperparameters['STRUCTURE']['n_MV']
    n_CV = hyperparameters['STRUCTURE']['n_CV']
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    hlszs = hyperparameters['STRUCTURE']['hlszs']

    model = NeuralNetwork(n_MV=n_MV, n_CV=n_CV, mu=mu, my=my, hlszs=hlszs)

    learning_rate = hyperparameters['TRAINING']['lr']
    batch_size = hyperparameters['TRAINING']['bsz']
    epochs = hyperparameters['TRAINING']['e']

    loss_fn = nn.MSELoss() # MSELoss since we're regressing; calculate error in cont. approx

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gen_data_config_path = Path(__file__).parent / "../config/generate_data.yaml"
    train_dl = load_stepresponse_data(csv_path_train, gen_data_config_path, train=True,
                                            bsz=batch_size, mu=mu, my=my, shuffle=True)
    
    train_dl = load_stepresponse_data(csv_path_train, gen_data_config_path, train=True,
                                            bsz=batch_size, mu=mu, my=my, shuffle=True)

    # train_dl, val_dl = load_stepresponse_data(csv_path_train, gen_data_config_path, train=True,
    #                                         bsz=batch_size, mu=mu, my=my, shuffle=True)

    val_dl = load_stepresponse_data(csv_path_val, gen_data_config_path, train=False,
                                            bsz=batch_size, mu=mu, my=my, shuffle=True)
    
    # test_dl = load_stepresponse_data(csv_path_test, gen_data_config_path, train=False,
    #                                         bsz=batch_size, mu=mu, my=my, shuffle=True)
    
    # ----- TRAINING ----- #
    train_losses = [0] * epochs
    # test_losses = [0] * epochs
    val_MSEs = [0] * epochs
    time = [t for t in range(epochs)]

    es = EarlyStopping()

    t = 0
    done = False
    while t < len(time) and not done:
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses[t] = train_loop(train_dl, model, loss_fn, optimizer)
        val_MSEs[t] = validation_loop(val_dl, model, t)
        
        # test_losses[t] = test_loop(test_dl, model, loss_fn)

        # Stop early, based on where the difference between validation and training errors is smallest
        # if es(abs(val_MSEs[t]-train_losses[t]), model):
        # Stopping early based on when validation error starts increasing.
        if es(abs(val_MSEs[t]), model):
            done = True
        
        t += 1

    model.log_MSE(val_MSEs[t-1])

    print("Done!")


    return model, train_losses, val_MSEs, time, t
    # return model, train_losses, val_MSEs, test_losses, time, t
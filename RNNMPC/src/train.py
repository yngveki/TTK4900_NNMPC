#!/usr/bin/env python3

# Trains neural network for SingleWell model estimation on some given dataset

import torch
import torch.nn as nn

from src.neuralnetwork import NeuralNetwork
from generate_data.load_input_data import load_input_data
from src.utils.custom_timing import Timer

class EarlyStopping():

    def __init__(self, p=5, min_delta=0, restore_best_weights=True):
        self.patience = p
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss, model):
        # initial call
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = model

        # validation loss decreased; update model because this one must be better
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())

        # validation loss increased; losing patience
        elif self.best_loss - val_loss <= self.min_delta:
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
    num_batches = len(dataloader)
    total_itrs = 0
    tot_loss = 0

    oldie = False
    if not oldie:
        for batch_nr, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = []
            outputs = []
            labels = []
            batch_mse = 0
            for u1, u2, y1, y2, truth in zip(batch['u1'], batch['u2'], batch['y1'], batch['y2'], batch['target']):
                X = torch.cat((u1.float(), u2.float(), y1.float(), y2.float()))
                output = model(X)
                truth = truth.float()

                # inputs.append(X)
                # outputs.append(output)
                # labels.append(truth)
                loss = loss_fn(output, truth)
                tot_loss += loss
                batch_mse += loss
            
            # TODO: Do we need to normalize? * (1/batch_size)? Even if it is taken care of automatically or not, it should be possible to tune lr around it
            batch_mse.backward() # Gradient of sum of this batch's losses find gradient of sum of all losses.
            optimizer.step()

            if batch_nr % 50 == 0:
                print(f"loss: {batch_mse.item():>9f},  [batch: {batch_nr:>5d}/{num_batches:>5d}]")


    else:
        for batch, sample in enumerate(dataloader.dataset):
            # Ensure zero-state from any previous iteration
            optimizer.zero_grad()

            u1 = torch.Tensor(sample['u1'])
            u2 = torch.Tensor(sample['u2'])
            y1 = torch.Tensor(sample['y1'])
            y2 = torch.Tensor(sample['y2'])
            X = torch.cat((u1, u2, y1, y2))
            truth = torch.Tensor(sample['target'])

            # Compute prediction and loss
            pred = model(X) 
            loss = loss_fn(pred, truth)

            tot_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch % 500 == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>9f}  [{current:>5d}/{size:>5d}]")
        
    total_itrs += 1
    
    return (tot_loss / total_itrs).item()

def validation_loop(dataloader, model, epoch):
    model.eval()
    mse_val = 0
    for sample in dataloader.dataset:
        # Prepare format of data
        u1 = torch.Tensor(sample['u1'])
        u2 = torch.Tensor(sample['u2'])
        y1 = torch.Tensor(sample['y1'])
        y2 = torch.Tensor(sample['y2'])
        X = torch.cat((u1, u2, y1, y2))
        truth = torch.Tensor(sample['target'])

        mse_val += torch.sum(torch.pow(truth - model(X), 2)).item()
    
    mse_val /= len(dataloader.dataset)
    print(f'Epoch: {epoch + 1}: Validation MSE: {mse_val}')
    return mse_val

# ----- TRAINING ----- #
# TODO: Plot norm of gradient over time to investigate whether local minima are the issue or not
def train(hyperparameters, csv_path_train, csv_path_val):

    act = hyperparameters['activation']
    mu = hyperparameters['STRUCTURE']['mu']
    my = hyperparameters['STRUCTURE']['my']
    hlszs = hyperparameters['STRUCTURE']['hlszs']
    learning_rate = hyperparameters['LEARNING']['lr']
    batch_size = hyperparameters['LEARNING']['bsz']
    epochs = hyperparameters['LEARNING']['e']
    patience = hyperparameters['LEARNING']['p']

    #! n_MV = 2 and n_CV = 2 are implicitly hardcoded here
    layers = []
    layers.append(2 * (mu + 1) + 2 * (my + 1)) # Input layer
    layers += hlszs                            # Hidden layers
    layers.append(2)                           # Output layer
    model = NeuralNetwork(layers=layers, act=act)
    loss_fn = nn.MSELoss() # MSE as loss func. for a regression problem
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dl = load_input_data(csv_path_train, bsz=batch_size, mu=mu, my=my, shuffle=True)
    val_dl = load_input_data(csv_path_val, bsz=batch_size, mu=mu, my=my, shuffle=True)
    
    # ----- TRAINING ----- #
    train_losses = [0] * epochs
    val_MSEs = [0] * epochs
    time = [t for t in range(epochs)]

    es = EarlyStopping(p=patience)

    t = 0
    done = False
    stopwatch = Timer()
    stopwatch.start()
    print_params = True
    while t < len(time) and not done:
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses[t] = train_loop(train_dl, model, loss_fn, optimizer)
        val_MSEs[t] = validation_loop(val_dl, model, t) # Necessary only for early stopping

        # Stopping early based on when validation error starts increasing.
        if es(abs(val_MSEs[t]), model):
            done = True
        
        t += 1
        stopwatch.lap()

        if print_params:
            for name, param in model.named_parameters():
                print(name, param.data)
    
    stopwatch.total_time()

    model.log_MSE(es.best_loss)

    print("Done!")

    return model, train_losses, val_MSEs, time, t
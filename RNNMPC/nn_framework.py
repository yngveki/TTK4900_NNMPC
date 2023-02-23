#!/usr/bin/env python3

import torch
from pathlib import Path
from os.path import exists
from os import makedirs
from yaml import safe_load, dump
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.neuralnetwork import NeuralNetwork
from src.train import train
from src.test import test
from src.utils.splitter import Splitter

class GroundTruth():
    """Holds arrays showing u1, u2, y1, y2"""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.u1 = np.asarray(df.iloc[0,:])
        self.u2 = np.asarray(df.iloc[1,:])
        self.y1 = np.asarray(df.iloc[2,:])
        self.y2 = np.asarray(df.iloc[3,:])

    def plot(self):
        _, axs = plt.subplots(2, 2)
        t = np.linspace(0, len(self) - 1, num=len(self))

        axs[0,0].plot(t, self.y1, label='gas rate', color='tab:orange')
        axs[0,0].legend()
        axs[1,0].plot(t, self.u1, label='choke')
        axs[1,0].legend()
        axs[0,1].plot(t, self.y2, label='oil rate', color='tab:orange')
        axs[0,1].legend()
        axs[1,1].plot(t, self.u2, label='GL rate')
        axs[1,1].legend()

        plt.show()

    def __len__(self):
        if all((len(self.u1) != len(self.u2), 
                len(self.u1) != len(self.y1), 
                len(self.u1) != len(self.y2))):
            Exception('All data sequences must be equally long!\n')

        return len(self.u1)

# ----- SCRIPT BEGINS ----- #
TEST = True

hyperparameter_name = 'config/nn_config.yaml'
hyperparameter_path = Path(__file__).parent / hyperparameter_name
with open(hyperparameter_path, "r") as f:
    hyperparameters = safe_load(f)

suffixes = ['.png', '.eps'] # Save formats for figures
model_nr = 1
model_name = "model_masteroppgave_" + str(model_nr)

# -- TRAINING -- #
# TODO: Make external loop to iterate over hyperparameter candidates
if __name__ == '__main__' and not TEST:
    # -- SETUP -- #
    csv_path_train = Path(__file__).parent / "generate_data/outputs/random_choke/csv/random_choke_4_output_clipped.csv"
    csv_path_val = Path(__file__).parent / "generate_data/outputs/random_choke/csv/random_choke_1_output_clipped.csv"

    # Saving which files were used for training and validation for traceability purposes
    hyperparameters['FILES'] = {'train_file': csv_path_train.__str__(), 'val_file': csv_path_val.__str__()}

    # -- TRAINING -- #              
    model, train_losses, val_MSEs, time, final_epoch = train(hyperparameters, csv_path_train, csv_path_val)

    # -- PLOTTING -- #
    p = hyperparameters['LEARNING']['p'] + 1 # To account for zero-indexing
    fig, ax = plt.subplots()

    # Plotting training against validation error
    fig.tight_layout()
    ax.plot(val_MSEs[1:final_epoch], 'r-', linewidth=2.0, label='Validation MSE')
    ax.plot(train_losses[1:final_epoch], 'b--', linewidth=2.0, label='Training losses')
    ax.axvline(len(val_MSEs[1:final_epoch]) - p, color='tab:red')
    ax.set_xlabel('epochs')
    ax.set_title(f'Validation performance over epochs. Lowest MSE: {model.mse:.3g}')
    ax.legend(loc='center right', prop={'size': 15})

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    plt.pause(1)
    plt.close()
    
    # -- SAVING TRAINED MODEL -- #
    # Setting up directory for results
    parent_dir = Path(__file__).parent / 'models'
    save_dir = parent_dir / model_name
    if not exists(save_dir):
        makedirs(save_dir)

    model_save_path = save_dir / (model_name + '.pt')
    yaml_save_path = save_dir / (model_name + '.yaml')

    if exists(model_save_path):
        name = input("Model with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
        if name != '': # _not_ aborting
            if name != 'y': # do _not_ want to override
                model_save_path = save_dir  / (name + '.pt')
                yaml_save_path = save_dir / (name + '_config' + '.yaml')

        else:
            model_save_path = None
            yaml_save_path = None
    
    if model_save_path is not None and yaml_save_path is not None:
        # Save model and corresponding config file used to generate that specific model
        torch.save(model.state_dict(), model_save_path)
        with open(yaml_save_path, "w", encoding = "utf-8") as yaml_file:
            yaml_file.write(dump(hyperparameters, default_flow_style = False, allow_unicode = True, encoding = None))
    else:
        print('Model was not saved')
        print(f'Status:\n\tmodel_save_path: {model_save_path}\n\tyaml_save_path: {yaml_save_path}')

    
    # -- SAVING FIGS -- #    
    save_dir /= 'figs'
    if not exists(save_dir):
        makedirs(save_dir)
    save_path = save_dir / 'trainval'

    if      exists(save_path.parent / (save_path.stem + '.png')) \
        or  exists(save_path.parent / (save_path.stem + '.eps')):
        name = input("Figure(s) with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
        if name != '': # _not_ aborting
            if name != 'y': # do _not_ want to override
                save_path = save_path.parent / name

        else:
            save_path = None
        
    if save_path is not None:
        for suffix in suffixes:
            save_path = save_path.parent / (save_path.stem + suffix)
            fig.savefig(save_path, bbox_inches='tight')
    else:
        print("Model was not saved.")

# -- TESTING -- #
if __name__ == '__main__' and TEST:
    # -- SETUP -- #
    csv_path_test = Path(__file__).parent / "generate_data/outputs/steps_choke/csv/step_choke_50_52_output_clipped.csv"
    model_path = Path(__file__).parent / ('models/' + model_name + '/' + model_name + '.pt')
    
    # -- PREDICTION -- #      
    gt = GroundTruth(csv_path_test)
    pred, model = test(model_path, csv_path_test, hyperparameters)
    # -- PLOTTING -- #
    fig2, axes = plt.subplots(2, 2, sharex=True)
    fig2.suptitle(f'Test MSE: {model.mse:.5g}', fontsize=23)

    # Plotting ground truth and predicted gas rates
    axes[0,0].set_title('Predicted v. true dynamics, gas rate', fontsize=20)
    axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
    axes[0,0].plot(gt.y1, '-', label='true gas rate', color='tab:orange')
    axes[0,0].plot(pred['y1'], label='predicted gas rate', color='tab:red')
    axes[0,0].legend(loc='best', prop={'size': 15})

    # Plotting ground truth and predicted oil rates
    axes[0,1].set_title('Predicted v. true dynamics, oil rate', fontsize=20)
    axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
    axes[0,1].plot(gt.y2, label='true oil rate', color='tab:orange')
    axes[0,1].plot(pred['y2'], '-', label='predicted oil rate', color='tab:red')
    axes[0,1].legend(loc='best', prop={'size': 15})

    # Plotting history of choke input
    axes[1,0].set_title('Input: choke', fontsize=20)
    axes[1,0].set_xlabel('time [s]', fontsize=15)
    axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
    axes[1,0].plot(gt.u1, label='choke', color='blue')
    axes[1,0].legend(loc='best', prop={'size': 15})

    # Plotting history of gas lift rate input
    axes[1,1].set_title('Input: gas lift rate', fontsize=20)
    axes[1,1].set_xlabel('time [s]', fontsize=15)
    axes[1,1].set_ylabel('percent opening [m^3/h]', fontsize=15)
    axes[1,1].plot(gt.u2, label='gas lift rate', color='blue')
    axes[1,1].legend(loc='best', prop={'size': 15})

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    plt.pause(1)
    plt.close()          

    
    # -- SAVING FIGS -- #
    parent_dir = Path(__file__).parent / ('models/' + model_name)
    save_dir = parent_dir / 'figs'
    if not exists(save_dir):
        temp = Path(__file__).parent / 'test'
        print(f'save_dir ({save_dir}) does not exist - default save to: {temp}')
        save_dir = temp
    save_path = save_dir / 'test'

    if      exists(save_path.parent / (save_path.stem + '.png')) \
        or  exists(save_path.parent / (save_path.stem + '.eps')):
        name = input("Figure(s) with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
        if name != '': # _not_ aborting
            if name != 'y': # do _not_ want to override
                save_path = save_path.parent / name

        else:
            save_path = None
        
    if save_path is not None:
        for suffix in suffixes:
            save_path = save_path.parent / (save_path.stem + suffix)
            fig2.savefig(save_path, bbox_inches='tight')
        
        txt_file = open(save_path.parent / (save_path.stem + '.txt'), 'a')
        txt_file.write('Test results showed in \'' + str(save_path.stem) + '\' came from testing with file at:\n\t' + str(csv_path_test) + '\n')
        txt_file.close()

    else:
        print("Test of model was not saved.")
#!/usr/bin/env python3

import torch
from pathlib import Path
from os.path import exists
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


if __name__ == '__main__':
    TEST = True
    nr = 4
    csv_path_train = Path(__file__).parent / "generate_data/data/normalized_u1_50_u2_7500_stairs_0_36000.csv"
    csv_path_val = Path(__file__).parent / "generate_data/data/normalized_u1_50_u2_7500_stairs_0_10000.csv"
    csv_path_test = Path(__file__).parent / "generate_data/data/normalized_u1_50_u2_7500_stairs_0_5000.csv"
    model_name = "model_prosjektoppgave_" + str(nr)
    model_path = Path(__file__).parent / "".join(("models/", model_name, ".pt"))
    yaml_save_path = Path(__file__).parent / "".join(("models/corresponding_config/", model_name, ".yaml"))

    hyperparameter_path = Path(__file__).parent / "config/nn_config.yaml"
    with open(hyperparameter_path, "r") as f:
        hyperparameters = safe_load(f)
        if not hyperparameters:
            Exception("Failed loading hyperparameters\n")

    hyperparameter_nr = "hyperparameters_" + str(nr)

# TODO: Make external loop to iterate over hyperparameter candidates
    if not TEST:
        # ----- TRAINING AND PLOTTING ----- #              
        model, train_losses, val_MSEs, time, final_epoch = train(hyperparameters, csv_path_train, csv_path_val)

        p = hyperparameters['LEARNING']['p'] + 1 # To account for zero-indexing
        
        # Plotting training against validation error
        fig, ax = plt.subplots()

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
        plt.pause(10)
        plt.close()
        
        # ----- SAVING FIGS AND TRAINED MODEL ----- #    
        fig_path_base = Path(__file__).parent
        fig_path_suffix = "figs/" + hyperparameter_nr + "_val.eps"
        fig.savefig(fig_path_base / fig_path_suffix, bbox_inches='tight')
        
        fig_path_suffix = "figs/" + hyperparameter_nr + "_val.png"
        fig.savefig(fig_path_base / fig_path_suffix, bbox_inches='tight')

        if not exists(model_path): # Safe to save; won't override
            torch.save(model.state_dict(), model_path)

            # Also save config file to have a record of what config was used to generate what model
            with open(yaml_save_path, "w", encoding = "utf-8") as yaml_file:
                deuce = dump(hyperparameters, default_flow_style = False, allow_unicode = True, encoding = None)
                yaml_file.write(deuce)
            
        else:
            decision = input("Model with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if decision != '':
                if decision != 'y':
                    model_path = Path(__file__).parent / "".join(("models/", decision, ".pt"))

                torch.save(model.state_dict(), model_path)

                # Also save config file to have a record of what config was used to generate what model
                with open(yaml_save_path, "w", encoding = "utf-8") as yaml_file:
                    deuce = dump(hyperparameters, default_flow_style = False, allow_unicode = True, encoding = None)
                    yaml_file.write(deuce)

            else:
                print("Model was not saved.")
    
    else:
        # ----- PREDICTION AND PLOTTING ----- #      
        gt = GroundTruth(csv_path_test)
        pred, model = test(model_path, csv_path_test, hyperparameters)

        fig2, axes = plt.subplots(2, 2, sharex=True)
        fig2.suptitle(f'Test MSE: {model.mse:.3g}', fontsize=23)

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
        plt.pause(30)
        plt.close()

        # ----- SAVING FIGS ----- #    
        fig_path_base = Path(__file__).parent
        fig_path_suffix = "figs/" + hyperparameter_nr + "_test.eps"
        fig2.savefig(fig_path_base / fig_path_suffix, bbox_inches='tight')
        
        fig_path_suffix = "figs/" + hyperparameter_nr + "_test.png"
        fig2.savefig(fig_path_base / fig_path_suffix, bbox_inches='tight')

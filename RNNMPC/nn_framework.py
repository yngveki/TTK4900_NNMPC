#!/usr/bin/env python3

import torch
from pathlib import Path
from os.path import exists
from os import makedirs
from yaml import safe_load, dump
from itertools import product
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

    def denormalize(self, u1_scale=None, u2_scale=None, y1_scale=None, y2_scale=None):
        if u1_scale is not None:
            self.u1 = [el * u1_scale for el in self.u1]
        if u1_scale is not None:
            self.u2 = [el * u2_scale for el in self.u2]
        if u1_scale is not None:
            self.y1 = [el * y1_scale for el in self.y1]
        if u1_scale is not None:
            self.y2 = [el * y2_scale for el in self.y2]

    def __len__(self):
        if all((len(self.u1) != len(self.u2), 
                len(self.u1) != len(self.y1), 
                len(self.u1) != len(self.y2))):
            Exception('All data sequences must be equally long!\n')

        return len(self.u1)

def grid_search_params(config_path):
    """
    Parses a config file with lists of candidate values for all parameters 
    into a list of parameter sets, grid search-style
    """
    
    def parse_to_dict(values, keys):
        """
        Parses an individual combination of parameter values back to dictionary format
        
        Note that \'m\' is implemented this way because mu and my must always be alike,
        so they cannot be implemented as two distinct lists in the config-file
        """
        d = {}
        for val, key in zip(values, keys):
            d[key] = val
        d['mu'] = d['m']
        d['my'] = d['m']
        del d['m']
        return d
    
    with open(config_path, "r") as f:
        params = safe_load(f)

    all_param_sets = []

    # Set up grid
    grid = product(*params.values())
    for point in grid:
        all_param_sets.append(parse_to_dict(point, params.keys()))

    return all_param_sets

# ----- SCRIPT BEGINS ----- #
# -- SETUP -- #
TRAIN = False
GRID = True
TEST = True

csv_path_train = Path(__file__).parent / 'generate_data/outputs/rnnmpc_random_walk/csv/random_walk_1mill_globally_normalized.csv'
csv_path_val = Path(__file__).parent / 'generate_data/outputs/rnnmpc_random_walk/csv/random_walk_50k_globally_normalized.csv'

tests = [(Path(__file__).parent / 'generate_data/outputs/steps_choke/csv/step_choke_50_52_globally_normalized.csv', 'step_choke_medium'),
         (Path(__file__).parent / 'generate_data/outputs/steps_choke/csv/step_choke_20_22_globally_normalized.csv', 'step_choke_low'),
         (Path(__file__).parent / 'generate_data/outputs/steps_choke/csv/step_choke_98_100_globally_normalized.csv', 'step_choke_high'),
         (Path(__file__).parent / 'generate_data/outputs/rnnmpc_random_walk/csv/random_walk_10k_globally_normalized.csv', 'random_walk_short'),
         (Path(__file__).parent / 'generate_data/outputs/random_mixed_ramp/csv/mixed_ramp_short_1_globally_normalized.csv', 'mixed_ramp_short'),
         (Path(__file__).parent / 'generate_data/outputs/ramp/csv/ramp_choke_gl_interval100_globally_normalized.csv', 'multiple_steps_interval100')]

model_nr_offset = 0
model_name = 'model_grid_'

delta_t = 10
suffixes = ['.png', '.eps'] # Save formats for figures

denormalize_test_results = True
denormalization_coeffs = {'u1_scale': (100-0), 'u2_scale': (10000-0), 'y1_scale': (18537-0), 'y2_scale': (349-0)}

plot_offset = True

# -- SETUP HYPERPARAMETERS -- #
if GRID == True:
    hyperparameter_name = 'config/nn_config_grid.yaml'
    hyperparameter_path = Path(__file__).parent / hyperparameter_name
    sets = grid_search_params(hyperparameter_path)
else:
    hyperparameter_name = 'config/nn_config.yaml'
    hyperparameter_path = Path(__file__).parent / hyperparameter_name
    with open(hyperparameter_path, "r") as f:
        hyperparameters = safe_load(f)
    sets = [hyperparameters]

# -- TRAINING -- #
if __name__ == '__main__' and TRAIN:
    # Train across all hyperparameters (only 1 set if GRID == False)
    for i, hyperparameters in enumerate(sets):
        model_name_train = model_name + str(model_nr_offset + i)

        # Saving which files were used for training and validation for traceability purposes
        hyperparameters['FILES'] = {'train_file': csv_path_train.__str__(), 'val_file': csv_path_val.__str__()}

        # -- TRAINING -- #              
        model, train_losses, val_MSEs, time, final_epoch = train(hyperparameters, csv_path_train, csv_path_val)

        # -- PLOTTING -- #
        p = hyperparameters['p'] + 1 # To account for zero-indexing
        fig, ax = plt.subplots()

        # Plotting training against validation error
        fig.tight_layout()
        ax.plot(val_MSEs[:final_epoch], 'r-', linewidth=2.0, label='Validation MSE')
        ax.plot(train_losses[:final_epoch], 'b--', linewidth=2.0, label='Training losses')
        ax.axvline(len(val_MSEs[:final_epoch]) - p + 1, color='tab:red') # +1 corrects for 0-indexing
        ax.set_xlabel('epochs')
        ax.set_title(f'Validation performance over epochs. Lowest MSE: {model.mse:.3g}')
        ax.legend(loc='center right', prop={'size': 15})
        ax.set_yscale('log')

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show(block=False)
        plt.pause(15)
        plt.close()
        
        # -- SAVING TRAINED MODEL -- #
        # Setting up directory for results
        parent_dir = Path(__file__).parent / 'models'
        save_dir = parent_dir / model_name_train
        if not exists(save_dir):
            makedirs(save_dir)

        model_save_path = save_dir / (model_name_train + '.pt')
        yaml_save_path = save_dir / (model_name_train + '.yaml')

        if exists(model_save_path):
            name = input("Model with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if name != '': # _not_ aborting
                if name != 'y': # do _not_ want to override
                    model_save_path = save_dir  / (name + '.pt')
                    yaml_save_path = save_dir / (name + '.yaml')

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
            print("Figures were not saved.")

# -- TESTING -- #
if __name__ == '__main__' and TEST:
    for i, hyperparameters in enumerate(sets): # (for each model as defined in all the hyperparameters)
        model_name_test = model_name + str(model_nr_offset + i)

        # -- SETUP -- #
        model_path = Path(__file__).parent / ('models/' + model_name_test + '/' + model_name_test + '.pt')
        if not exists(model_path):
            print(f'no model at path: {model_path}')
            continue

        hyperparameter_path = model_path.parent / (model_path.stem + '.yaml')
        with open(hyperparameter_path, "r") as f:
            hyperparameters = safe_load(f)

        # -- PREDICTION -- # (for each test defined in tests)     
        for csv_path_test, test_save_name in tests:
            gt = GroundTruth(csv_path_test)
            pred, model = test(model_path, csv_path_test, hyperparameters)

            if denormalize_test_results:
                gt.denormalize(**denormalization_coeffs)
                pred['y1'] = [el * denormalization_coeffs['y1_scale'] for el in pred['y1']]
                pred['bias y1'] = [el * denormalization_coeffs['y1_scale'] for el in pred['bias y1']]
                pred['y2'] = [el * denormalization_coeffs['y2_scale'] for el in pred['y2']]
                pred['bias y2'] = [el * denormalization_coeffs['y2_scale'] for el in pred['bias y2']]

            # -- PLOTTING -- #
            fig2, axes = plt.subplots(2, 2, sharex=True)
            fig2.suptitle(f'Test MSE: {model.mse:.5g}', fontsize=23)

            # predictions are offset because of mu and my. Compensation:
            t = np.linspace(0, delta_t * len(gt.y1), num=len(gt.y1))
            offset_y1 = len(gt.y1) - len(pred['y1'])
            offset_y2 = len(gt.y2) - len(pred['y2'])

            # Plotting ground truth and predicted gas rates
            axes[0,0].set_title('Predicted v. true dynamics, gas rate', fontsize=20)
            axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
            axes[0,0].plot(t, gt.y1, '-', label='true gas rate', color='tab:orange')
            axes[0,0].plot(t[offset_y1:], pred['y1'], label='predicted gas rate', color='tab:red')
            axes[0,0].legend(loc='best', prop={'size': 15})
            # axes[0,0].set_ylim(0, denormalization_coeffs['y1_scale'] * 1.1)

            if plot_offset:
                axes00_twinx = axes[0,0].twinx()
                axes00_twinx.plot(t[offset_y1:], pred['bias y1'], '--', linewidth=0.3, color='tab:green')
                axes00_twinx.set_ylabel('diff. ground truth v. predicted gas rate [m^3/h]', color='tab:green')
                axes00_twinx.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
                axes00_twinx.spines['right'].set_color('tab:green')
                max_lim = max(pred['bias y1']) + (2.5 * abs(max(pred['bias y1'])))
                min_lim = min(pred['bias y1']) - (0.5 * abs(min(pred['bias y1'])))
                axes00_twinx.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

            # Plotting ground truth and predicted oil rates
            axes[0,1].set_title('Predicted v. true dynamics, oil rate', fontsize=20)
            axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
            axes[0,1].plot(t, gt.y2, label='true oil rate', color='tab:orange')
            axes[0,1].plot(t[offset_y2:], pred['y2'], '-', label='predicted oil rate', color='tab:red')
            axes[0,1].legend(loc='best', prop={'size': 15})
            # axes[0,1].set_ylim(0, denormalization_coeffs['y2_scale'] * 1.1)
            
            if plot_offset:
                axes01_twinx = axes[0,1].twinx()
                axes01_twinx.plot(t[offset_y2:], pred['bias y2'], '--', linewidth=0.3, color='tab:green')
                axes01_twinx.set_ylabel('diff. ground truth v. predicted oil rate [m^3/h]', color='tab:green')
                axes01_twinx.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
                axes01_twinx.spines['right'].set_color('tab:green')
                max_lim = max(pred['bias y2']) + (2.5 * abs(max(pred['bias y2'])))
                min_lim = min(pred['bias y2']) - (0.5 * abs(min(pred['bias y2'])))
                axes01_twinx.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

            # Plotting history of choke input
            axes[1,0].set_title('Input: choke', fontsize=20)
            axes[1,0].set_xlabel('time [s]', fontsize=15)
            axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
            axes[1,0].plot(t, gt.u1, label='choke', color='blue')
            axes[1,0].legend(loc='best', prop={'size': 15})

            # Plotting history of gas lift rate input
            axes[1,1].set_title('Input: gas lift rate', fontsize=20)
            axes[1,1].set_xlabel('time [s]', fontsize=15)
            axes[1,1].set_ylabel('percent opening [m^3/h]', fontsize=15)
            axes[1,1].plot(t, gt.u2, label='gas lift rate', color='blue')
            axes[1,1].legend(loc='best', prop={'size': 15})

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

            plt.show(block=False)
            plt.pause(1)
            plt.close()          
            
            # -- SAVING FIGS -- #
            parent_dir = Path(__file__).parent / ('models/' + model_name_test)
            save_dir = parent_dir / 'figs'
            if not exists(save_dir):
                temp = Path(__file__).parent / test_save_name
                print(f'save_dir ({save_dir}) does not exist - default save to: {temp}')
                save_dir = temp
            save_path = save_dir / test_save_name

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
                
                txt_file = open(save_path.parent / ('tests.txt'), 'a')
                txt_file.write('Test results showed in \'' + str(save_path.stem) + '\' came from testing with file at:\n\t' + str(csv_path_test) + '\n')
                txt_file.close()

            else:
                print("Test of model was not saved.")
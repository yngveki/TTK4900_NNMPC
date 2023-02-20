# Generates csv files describing sequences of input-data
# coherent with the class Timeseries, defined in 
# TTK4900_NNMPC/src/utils/timeseries

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from pathlib import Path
from matplotlib.pylab import plt
import csv
from os.path import exists
from yaml import safe_load

from src.utils.timeseries import Timeseries

def safe_save(csv_path, data):
    assert isinstance(data, list), "This save function is implemented only for lists of csv-rows!"

    if not exists(csv_path):    # Safe to save; nothing can be overwritten
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            for row in data:
                writer.writerow(row)
    else:
        name = input("File already exists. Provide new name or \'y\' to overwrite ([enter] aborts. File-endings are automatic!): ")
        if name != '': # Do _not_ abort save
            if name != 'y': # Do _not_ overwrite
                csv_path = Path(__file__).parent / "data/steps" / ''.join((name, '.csv'))
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                for row in data:
                    writer.writerow(row)

            print(f"File written to \'{csv_path}\'")

        else:
            print("File was not saved.")

def step_csv(start_time, step_time, final_time, init_val, step_val):
    """
    Defines a list of rows that together make up a step, when using
    the Timeseries class to represent the csv-file
    
    args:
        :param start_time: start time of the sequence
        :param step_time: time value at moment of step
        :param final_time: final time of the sequence
        :param init_val: initial value before step
        :param step_val: value after the step (note that this is not the increment-value)
    """

    sequence = []
    sequence.append(['time', 'choke', 'GL']) # Header
    sequence.append([start_time, init_val[0], init_val[1]])
    sequence.append([step_time, step_val[0], step_val[1]])
    sequence.append([final_time, step_val[0], step_val[1]])

    return sequence

if __name__ == '__main__':
    # Generating lots of steps
    stepped_input = 'GL'
    reverse = True

    if stepped_input == 'choke':
        bounds = [0,100]
        resolution = 2 # choke changes incrementally with 2 [%] at a time
        init = [bounds[0],0]
        increment = [resolution, 0]
    elif stepped_input == 'GL':
        bounds = [2000,10000]
        resolution = 166 # gas lift changes incrementally ~166 [m^3/h] at a time
        init = [100,bounds[0]] # choke kept fully open, since only then do we actually want to use GL
        increment = [0, resolution]

    if reverse:
        resolution = -resolution
        increment = [-increment[i] for i in range(len(increment))]
        bounds = bounds[-1::-1]
        if stepped_input == 'choke':
            init[0] = bounds[0]
        elif stepped_input == 'GL':
            init[1] = bounds[0]

    start_time = 0
    step_time = 1000
    final_time = 5000

    for step in range(bounds[0], bounds[1], resolution):
        # Generate data
        step_val = [init[i] + increment[i] for i in range(len(init))]
        sequence = step_csv(start_time, step_time, final_time, init, step_val)

        # Save data
        if stepped_input == 'choke':
            filename = 'step_choke_' + str(init[0]) + '_' + str(step_val[0]) + '.csv'
            csv_path = Path(__file__).parent / 'steps_choke/' / filename
        elif stepped_input == 'GL':
            filename = 'step_GL_' + str(init[1]) + '_' + str(step_val[1]) + '.csv'
            csv_path = Path(__file__).parent / 'steps_GL/' / filename
        
        safe_save(csv_path, sequence)

        # Update before next iteration
        init = step_val
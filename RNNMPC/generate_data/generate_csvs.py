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

def general_csv(data, time=0, delta_t=10, interval=10):
    """
    Defines a list of rows that together make up a series of steps
    when using the Timeseries class to represent the csv-file
    
    args:
        :param data: a 2d list containing: [[input_choke], [input_GL]]
        :param time: the first timestamp
        :param delta_t: length of each timestep
        :param interval: how many time steps should be enforced between each entry in data
    """
    sequence = []
    sequence.append(['time', 'choke', 'GL']) # Header

    assert isinstance(data, list), 'data must be a list of input for choke and GL'
    data_transposed = np.array(data).T
    for entry in data_transposed: # list(zip(.)) to transpose simple Python list without np or pd
        t1 = entry[0]
        t2 = entry[1]
        sequence.append([time, entry[0], entry[1]])
        time += delta_t * interval

    return sequence

def step(start_time, stop_time, step_time, delta_t=1, start_val=0, step_val=1):
    start = start_time // delta_t
    stop = stop_time // delta_t
    step = step_time // delta_t
    timeline = np.linspace(start, stop + 1, num=stop-start)
    input_profile = start_val + step_val * np.heaviside(timeline - step, 1)

    return input_profile

def staircase(init=0, lb=0, ub=100, increment=1, interval=1, num=50, lock_at=None):
    assert init >= lb and init <= ub, f'init must be within legal range!\n'

    signal = [0] * num
    signal[0] = init
    locked_at = None
    for t, sig in enumerate(signal):
        if t == 0:
            continue

        if t % 1000 == 0:
            print(f'iter: {t}')

        if (t % interval != 0):
            signal[t] = signal[t-1]
        else:        
            do_inc = np.random.choice([True, False])
            
            if do_inc:
                signal[t] = np.min((signal[t-1] + increment, ub)) 
            else:
                signal[t] = np.max((signal[t-1] - increment, lb))

        if lock_at is not None:
            if signal[t] == lock_at:
                signal[t:] = lock_at
                locked_at = t

    return signal, locked_at

def flatline(start_time, stop_time, step_time, delta_t=1, flat_val=1):
    """
    Convenience wrapper of the step-function to create a flat-line of adequate sizes
    """
    return step(start_time, stop_time, step_time, delta_t=delta_t, start_val=flat_val, step_val=0)
        
# ----- SCRIPT ----- #
sequence = 'staircase'

# -- Generating a random "staircase" type input_profile represented by a csv -- #
# the two input profiles are generated differently because we don't want to use GL before choke is maxed out
if __name__ == '__main__' and sequence == 'staircase':
    num = 10000
    input_choke, _ = staircase(init=0, lb=0, ub=100, increment=2, interval=1, num=num)
    input_GL, _ = staircase(init=2000, lb=2000, ub=10000, increment=166, interval=1, num=num)

    sequence = general_csv([input_choke, input_GL], time=0, delta_t=10, interval=10)
        
    filename = 'staircases_' + str(num) + '_steps.csv'
    csv_path = Path(__file__).parent / 'inputs/staircases' / filename
    safe_save(csv_path, sequence)

# -- Generating a set of unit steps in different inputs -- #
if __name__ == '__main__' and sequence == 'steps':
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

    for s in range(bounds[0], bounds[1], resolution):
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
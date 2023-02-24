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
from yaml import dump

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
    for entry in data_transposed:
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
        
def ramp_choke(min=0, max=100, resolution=2):
    """
    1) Kun choke - null gassløft, ramp choke opp fra 0% til 100% og ned til 0% med 2% sprang, der du venter 5 min mellom hver endring.
    
    args:
        :param min: minimum actuation for choke
        :param max: maximum actuation for choke
        :param resolution: how much choke changes for each step

    Note! Must be fed through "general_csv" with delta_t=10 and interval=30 to satisfy format for above specification!
    """
    choke = [i for i in range(min, max, resolution)] + [i for i in range(max, min - 1, (-resolution))] # include 0 as final value

    GL = [0] * (1 + 2 * (max - min) // resolution)
    return [choke, GL]

def random_choke(init=30, choke_bounds=[30,70], waiting_limits=[10,50], increment=2, num_steps=1000):
    """
    2) Kun choke - null gassløft, ramp choke opp og ned litt tilfeldig i typisk arbeidsområde 30-70%, varier hvor lenge du venter mellom hvert 2% sprang.
    
    args:
        :param init: initial value for choke
        :param choke_bounds: legal bounds for values for choke
        :param waiting_limits: bounds for random waiting time [timesteps]
        :param increment: how much choke changes for each step
        :param num_steps: how many steps should be performed during the simulation (_not_ how many timesteps)

    Note! Current implementation yields a simulation that is exactly num_steps amount of timesteps, since the random increments
          are not furthered to general_csv() later!
    """
    assert init >= choke_bounds[0] and init <= choke_bounds[1], f'init must be within legal range!\n'

    choke = [0] * num_steps
    choke[0] = init
    interval = np.random.randint(waiting_limits[0], waiting_limits[1])
    wait = interval - 1 # Since t == 0 causes one skipped decrement
    for t in range(len(choke)):
        if t == 0:
            continue

        if t % 100 == 0:
            print(f'iter: {t}')

        if wait == 0:
            do_inc = np.random.choice([True, False])
            
            if do_inc: # Increment
                choke[t] = np.min((choke[t-1] + increment, choke_bounds[1]))
            else:      # Decrement
                choke[t] = np.max((choke[t-1] - increment, choke_bounds[0]))

            interval = np.random.randint(waiting_limits[0], waiting_limits[1])
            wait = interval 
        else:
            choke[t] = choke[t - 1]
        
        wait -= 1

    GL = [0] * num_steps
    return [choke, GL]

# ----- SCRIPT ----- #
sequence = 'random_choke'

# -- Generating a randomly varying choke; similar to staircases, but more specified -- #
if __name__ == '__main__' and sequence == 'random_choke':
    for i in range(1):
        specifications = {'init': 50, 
                        'choke_bounds': [30,70], 
                        'waiting_limits': [15,35], # Should wait between 10 and 50 timesteps, hence interval=1 in general_csv below 
                        'increment': 2, 
                        'num_steps': (i+1) * 500}
        sequence = random_choke(**specifications)
        sequence = general_csv(sequence, interval=1)

        filename = 'random_choke'
        # nr = i
        csv_path = Path(__file__).parent / ('inputs/random_choke/' + filename + '_' + 'short' + '.csv')
        safe_save(csv_path, sequence)
        yaml_path = csv_path.parent / (csv_path.stem + '.yaml')
        with open(yaml_path, "w", encoding = "utf-8") as yaml_file:
            yaml_file.write(dump(specifications, default_flow_style = False, allow_unicode = True, encoding = None))

# -- Generating a controlled ramp from 0% to 100% to 0% opening in choke -- #
if __name__ == '__main__' and sequence == 'ramp_choke':
    sequence = ramp_choke()
    sequence = general_csv(sequence, time=0, delta_t=10, interval=30)
    csv_path = Path(__file__).parent / 'inputs/ramp_choke/ramp_choke_step2_interval30.csv'
    safe_save(csv_path, sequence)

# -- Generating a random "staircase" type input_profile represented by a csv -- #
# the two input profiles are generated differently because we don't want to use GL before choke is maxed out
if __name__ == '__main__' and sequence == 'staircase':
    num = 10
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
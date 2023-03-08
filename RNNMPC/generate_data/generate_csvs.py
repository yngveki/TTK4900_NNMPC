# Generates csv files describing sequences of input-data
# coherent with the class Timeseries, defined in 
# TTK4900_NNMPC/src/utils/timeseries

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.pylab import plt
import csv
from os.path import exists
from yaml import dump

from src.utils.timeseries import Timeseries

def safe_save(csv_path, data):
    assert isinstance(data, list), "This save function is implemented only for csv-rows of datatype `list`!"

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
        
def ramp(min, max, resolution):
    arr = [i for i in range(min, max, resolution)]
    if max not in arr: arr.append(max) # from (including) to (excluding) causes max element to be left out. I want it to be with
    return arr

def ramp_choke(min=0, max=100, resolution=2):
    """
    1) Kun choke - null gassløft, ramp choke opp fra 0% til 100% og ned til 0% med 2% sprang, der du venter 5 min mellom hver endring.
    
    args:
        :param min: minimum actuation for choke
        :param max: maximum actuation for choke
        :param resolution: how much choke changes for each step

    Note! Must be fed through "general_csv" with delta_t=10 and interval=30 to satisfy format for above specification!
    """
    arr1 = ramp(min, max, resolution)
    arr2 = ramp(max, min - 1, (-resolution))
    temp_choke = arr1 + arr2
    choke = [i for i in range(min, max, resolution)] + [i for i in range(max, min - 1, (-resolution))] # include 0 as final value

    assert temp_choke == choke, 'damn. Debugging time'
    GL = [0] * (1 + 2 * (max - min) // resolution)
    return [choke, GL]

def input_random_ramp(init=30, bounds=[30,70], waiting_limits=[10,50], increment=2, num_steps=1000, p_inc=[0.5,0.5], do_choke=True):
    """
    Randomly varying within given limits for specified input variable
    
    args:
        :param init: initial value for input variable
        :param bounds: legal bounds for values for input variable
        :param waiting_limits: bounds for random waiting time [timesteps]
        :param increment: how much input variable changes for each step
        :param num_steps: how many steps should be performed during the simulation (_not_ how many timesteps)
        :param p_inc: probabilistic distribution for incrementing or decrementing
        :param do_choke: whether choke is the randomly ramped input or not (`False` indicates gas lift instead)

    Note! Current implementation yields a simulation that is exactly num_steps amount of timesteps, since the random increments
          are not furthered to general_csv() later!
    """
    assert init >= bounds[0] and init <= bounds[1], f'init must be within legal range!\n'
    assert np.sum(p_inc) == 1.0, 'Probabilities must sum to 1.0 exactly!'

    inpt = [0] * num_steps
    inpt[0] = init
    interval = np.random.randint(waiting_limits[0], waiting_limits[1])
    wait = interval - 1 # Since t == 0 causes one skipped decrement
    for t in range(len(inpt)):
        if t == 0:
            continue

        if t % 100 == 0:
            print(f'iter: {t}')

        if wait == 0:
            do_inc = np.random.choice([True, False], p=p_inc)
            
            if do_inc: # Increment
                inpt[t] = np.min((inpt[t-1] + increment, bounds[1]))
            else:      # Decrement
                inpt[t] = np.max((inpt[t-1] - increment, bounds[0]))

            interval = np.random.randint(waiting_limits[0], waiting_limits[1])
            wait = interval 
        else:
            inpt[t] = inpt[t - 1]
        
        wait -= 1

    if do_choke:
        GL = [0] * num_steps
        return [inpt, GL]
    else:
        choke = [100] * num_steps
        return [choke, inpt]
    
def concatenate(*args):
    """
    Concatenates data given by csv-paths in args. Fragile! Assumes user gives only paths to files on same format!
    """
    for arg in args:
        assert isinstance(arg, Path), 'All arguments must represent a path to a .csv-file!'
    assert len(args) >= 2, 'Nothing to concatenate when fewer than 2 arguments!'

    df = pd.read_csv(args[0]) # read paths[0]
    for nr in range(1, len(args)):
        # Update timestamps
        t = df.values[-1][0]
        df_next = pd.read_csv(args[nr])
        for idx in range(len(df.values)):
            df_next.values[idx][0] += t

        # Extend dataframe
        df = df.append(df_next)

    sequence = df.values.tolist()
    sequence = []
    sequence.append(['time', 'choke', 'GL']) # Header
    sequence.extend(df.values.tolist()) # Stupid to have to do it like this, but I'd have to refactor more than I have time to if not using safe_save

    return sequence

# ----- SCRIPT ----- #
sequence = 'concatenate'

if __name__ == '__main__' and sequence == 'concatenate':
    paths = [Path(__file__).parent / 'inputs/steps_choke/step_choke_0_2.csv',
             Path(__file__).parent / 'inputs/steps_GL/step_GL_2000_2166.csv']
    # paths = [Path(__file__).parent / 'inputs/random_choke_ramp/random_choke_ramp_1.csv',
    #          Path(__file__).parent / 'inputs/ramp_choke/ramp_choke_step2_interval30_60-100.csv',
    #          Path(__file__).parent / 'inputs/ramp_gl/ramp_gl_step200_interval30_0-2000.csv',
    #          Path(__file__).parent / 'inputs/random_gl_ramp/random_gl_ramp_0.csv']
    
    concatenated = concatenate(*paths)
    save_path = Path(__file__).parent / 'inputs/random_mixed_ramp/mixed_test_0.csv'
    safe_save(save_path, concatenated)

if __name__ == '__main__' and sequence == 'random_gl_ramp':
    # TODO: Make p_inc = [100,0] for when beneath 2000
    global_min = 2000 # Assume already spun up by a ramp; concatenate input profiles later
    global_max = 10000    
    specifications = {'init': global_min, 
                    'bounds': [2000, 4000], 
                    'waiting_limits': [74,75], 
                    'increment': 200, 
                    'num_steps': 1000,
                    'p_inc': [0.55,0.45],
                    'do_choke': False}
    sequence = input_random_ramp(**specifications)
    rising = True    
    for i in range(99):
        # if i == 0:
        #     sequence.append(input_random_ramp(**specifications))
        # extension = input_random_ramp(**specifications)
        # if not len(sequence): # So far, it's empty
        #     sequence = extension
        # else:
        extension = input_random_ramp(**specifications)
        sequence[0].extend(extension[0])
        sequence[1].extend(extension[1])

        # Update for next iteration
        final_inpt_val = sequence[1][-1]
        if final_inpt_val == global_max: # Capped upwards, can span downwards again
            rising = False
        elif final_inpt_val == 2000: # Don't want to be working beneath 2000 when it's active
            rising = True

        if rising:
            specifications['p_inc'] = [0.53,0.47]
            specifications['bounds'] = [max(final_inpt_val, global_min), min(final_inpt_val + 2000, global_max)]
        else:
            specifications['p_inc'] = [0.47,0.53]
            specifications['bounds'] = [max(final_inpt_val - 2000, global_min), min(final_inpt_val, global_max)]
        specifications['init'] = final_inpt_val
    
    # Visualize result
    fig, axs = plt.subplots(2)
    axs[0].plot(sequence[0][:], label='choke', color='tab:orange')
    axs[0].legend(loc='best')
    axs[1].plot(sequence[1][:], label='gas_rate', color='tab:green')
    axs[1].legend(loc='best')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(block=False)
    plt.pause(15)
    plt.close()

    sequence = general_csv(sequence, interval=1)

    filename = 'random_gl_ramp'
    nr = 0
    csv_path = Path(__file__).parent / ('inputs/random_gl_ramp/' + filename + '_' + str(nr) + '.csv')
    safe_save(csv_path, sequence)
    yaml_path = csv_path.parent / (csv_path.stem + '.yaml')
    with open(yaml_path, "w", encoding = "utf-8") as yaml_file:
        yaml_file.write(dump(specifications, default_flow_style = False, allow_unicode = True, encoding = None))

if __name__ == '__main__' and sequence == 'random_choke_ramp':
    sequence = []
    global_min = 60
    global_max = 100
    specifications = {'init': global_min, 
                    'bounds': [global_min, 70], 
                    'waiting_limits': [74,75], 
                    'increment': 2, 
                    'num_steps': 1000,
                    'p_inc': [0.55,0.45]} # 55% chance to increment, 45% chance to decrement. Intentionally skew the ramp upwards.
    rising = True
    for i in range(10):
        extension = input_random_ramp(**specifications)
        if not len(sequence): # So far, it's empty
            sequence = extension
        else:
            sequence[0].extend(extension[0])
            sequence[1].extend(extension[1])

        # Update for next iteration
        final_inpt_val = sequence[0][-1]
        if final_inpt_val == global_max: # Capped upwards, can span downwards again
            rising = False
        elif final_inpt_val == global_min: # Capped downwards, can span upwards again. Also don't want to go below 60 after rising above
            rising = True

        if rising:
            specifications['p_inc'] = [0.53,0.47]
            specifications['bounds'] = [max(final_inpt_val, global_min), min(final_inpt_val + 20, global_max)]
        else:
            specifications['p_inc'] = [0.47,0.53]
            specifications['bounds'] = [max(final_inpt_val - 20, global_min), min(final_inpt_val, global_max)]
        specifications['init'] = final_inpt_val
    
    # Visualize result
    fig, axs = plt.subplots(2)
    axs[0].plot(sequence[0][:], label='choke', color='tab:orange')
    axs[0].legend(loc='best')
    axs[1].plot(sequence[1][:], label='gas_rate', color='tab:green')
    axs[1].legend(loc='best')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show(block=False)
    plt.pause(15)
    plt.close()

    sequence = general_csv(sequence, interval=1)

    filename = 'random_choke_ramp_short'
    nr = 2
    csv_path = Path(__file__).parent / ('inputs/random_choke_ramp/' + filename + '_' + str(nr) + '.csv')
    safe_save(csv_path, sequence)
    yaml_path = csv_path.parent / (csv_path.stem + '.yaml')
    with open(yaml_path, "w", encoding = "utf-8") as yaml_file:
        yaml_file.write(dump(specifications, default_flow_style = False, allow_unicode = True, encoding = None))

# -- Generate a steady slope to max out first choke, then gas lift -- #
if __name__ == '__main__' and sequence == 'max_slopes':
    # This dataset will not be used for training, just to determine max values for normalization of gas rate and oil rate
    choke = []
    choke.extend(ramp(0, 100, 2))
    gl = [0] * len(choke)
    gl.extend(ramp(0,10000, 200))
    gl.extend([10000 for _ in range(500)]) # Let's give it some time to settle
    choke.extend([100 for _ in range(len(gl)-len(choke))])

    assert len(choke) == len(gl)

    sequence = [choke, gl]
    sequence = general_csv(sequence, interval=10)
    csv_path = Path(__file__).parent / ('inputs/div/ramp_choke_and_gl.csv')
    safe_save(csv_path, sequence)

# -- Generating a randomly varying choke; similar to staircases, but more specified -- #
if __name__ == '__main__' and sequence == 'random_choke':
    for i in range(2,3):
        specifications = {'init': 50, 
                        'choke_bounds': [30,100], 
                        'waiting_limits': [99,100], 
                        'increment': 2, 
                        'num_steps': 1000}
        sequence = input_random_ramp(**specifications)
        sequence = general_csv(sequence, interval=1)

        filename = 'random_choke'
        nr = i
        csv_path = Path(__file__).parent / ('inputs/random_choke/' + filename + '_' + 'short_' + str(i) + '.csv')
        safe_save(csv_path, sequence)
        yaml_path = csv_path.parent / (csv_path.stem + '.yaml')
        with open(yaml_path, "w", encoding = "utf-8") as yaml_file:
            yaml_file.write(dump(specifications, default_flow_style = False, allow_unicode = True, encoding = None))

# -- Generating a controlled ramp for gas lift rate -- #
if __name__ == '__main__' and sequence == 'ramp_gl':
    GL = ramp(min=0, max=2000, resolution=200)
    choke = [100] * len(GL)
    sequence = general_csv([choke, GL], time=0, delta_t=10, interval=30)
    csv_path = Path(__file__).parent / 'inputs/ramp_gl/ramp_gl_step200_interval30_0-2000.csv'
    safe_save(csv_path, sequence)

# -- Generating a controlled ramp for choke opening -- #
if __name__ == '__main__' and sequence == 'ramp_choke':
    choke = ramp(min=60, max=100, resolution=2)
    GL = [0] * len(choke)
    sequence = general_csv([choke, GL], time=0, delta_t=10, interval=30)
    csv_path = Path(__file__).parent / 'inputs/ramp_choke/ramp_choke_step2_interval30_60-100.csv'
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
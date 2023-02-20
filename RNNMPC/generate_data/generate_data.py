# Generates data from SingleWell FMU for training of neural network

# For further reference on dealing with the FMU:
# https://jmodelica.org/pyfmi/tutorial.html#

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pyfmi import load_fmu
import numpy as np
from pathlib import Path
from matplotlib.pylab import plt
import csv
from os.path import exists
from yaml import safe_load

from src.utils.timeseries import Timeseries

def init_model(model_path, start_time, final_time, Uk=[[50],[0]], delta_t=10, warm_start_t=0, warm_start=False):
    # Initiates a FMU model at the given model_path
    #
    # Input:
    # - model_path:     path to relevant FMU
    # - start_time:     simulation start time
    # - final_time:     simulation final time
    # - warm_start_t:   for how long to perform warm start of simulation (must be integer divisible by delta_t)
    # - delta_t:        time step of system
    #
    # Returns: Properly initiated model

    time = start_time
    model = load_fmu(model_path, log_level=7) #loglevel 7: log everything

    #Set SepticControl --> True
    model.set_boolean([536871484], [True])

    model.initialize(start_time, final_time, True) #Initialize the slave

    y1 = []
    y2 = []

    if warm_start:
        assert isinstance(Uk, list), "Uk must be a list containing 2 elements only like so: Uk = [choke, gas lift]"
        for i in range(warm_start_t // delta_t):
            model.set_real([3,4], Uk)
            model.do_step(time, delta_t)

            time += delta_t
            y1.append(float(model.get('y[1]')))
            y2.append(float(model.get('y[2]')))

    return model, y1, y2

def simulate_singlewell_step(model, time, warm_start_t, delta_t, Uk, time_series=None):
    # Simulates one step of the SingleWell model, from time to time + delta_t.
    #
    # Inputs:
    # - model:          FMU in the form of a model object
    # - time:           current timestep from which we simulate 1 step ahead
    # - delta_t:        time interval per timestep
    # - warm_start_t:   for how long the simulation warm started during initialization
    # - delta_t:        time step of system
    # - Uk:             vector of actuations for current time step, shape: n_MV x 1
    # - time_series:    a time series that tracks for which timesteps we have simulated. For plotting purposes. (OPTIONAL)
    #
    # Returns: the two CVs of interest from the model. Optionally also the updated time series.
    
    # Apply input
    model.set_real([3,4], [Uk[0], Uk[1]]) # indices of inputs are retrieved from .xml of FMU when unpacked

    # Perform 1-step simulation
    time_offset = warm_start_t # * delta_t
    model.do_step(time, delta_t) # Perform 1-step simulation of delta_t from time

    # Get output
    gas_rate = float(model.get('y[1]'))
    oil_rate = float(model.get('y[2]'))
    
    return gas_rate, oil_rate

def step(start_time, stop_time, step_time, delta_t=1, start_val=0, step_val=1):
    start = start_time // delta_t
    stop = stop_time // delta_t
    step = step_time // delta_t
    timeline = np.linspace(start, stop + 1, num=stop-start)
    input_profile = start_val + step_val * np.heaviside(timeline - step, 1)

    return input_profile

def staircase(init=0, lb=0, ub=100, increment=1, interval=1, num=50):
    assert init >= lb and init <= ub, f'init must be within legal range!\n'

    signal = [0] * num
    signal[0] = init
    for t, sig in enumerate(signal):
        if t == 0:
            continue

        if (t % interval != 0):
            signal[t] = signal[t-1]
        else:        
            do_inc = np.random.choice([True, False])
            
            if do_inc:
                signal[t] = np.min((signal[t-1] + increment, ub)) 
            else:
                signal[t] = np.max((signal[t-1] - increment, lb))

    return signal

def flatline(start_time, stop_time, step_time, delta_t=1, flat_val=1):
    """
    Convenience wrapper of the step-function to create a flat-line of adequate sizes
    """
    return step(start_time, stop_time, step_time, delta_t=delta_t, start_val=flat_val, step_val=0)

def normalize(series, min=None, max=None):
    normalized = []
    if min == None:
        series_min = np.min(series)
    else:
        series_min = min
        
    if max == None:
        series_max = np.max(series)
    else:
        series_max = max

    for element in series:
        normalized_item = (element - series_min) / (series_max - series_min)
        normalized.append(normalized_item)

    return normalized

def clip_beginning(series, clip_length=100):
    return series[clip_length:]

# ----- SCRIPT RUN ----- #
if __name__ == '__main__':

    # ----- SETUP ----- #
    config_path = Path(__file__).parent / "../config/generate_data.yaml"
    with open(config_path, "r") as f:
        config = safe_load(f)
        if not config:
            Exception("Failed loading config file\n")

    warm_start_t = config['warm_start_t']
    delta_t = config['delta_t']
    resolution = -config['resolution']
    
    # -- Iterate over all csvs
    for i in range(10000,2000,resolution):
        filename = 'step_GL_' + str(i) + '_' + str(i + resolution)
        filepath = Path(__file__).parent / "steps_GL/" / ''.join((filename, '.csv'))
        input_profile = Timeseries(filepath, delta_t=10)
        start_time = input_profile.begin
        final_time = input_profile.end

        warm_start_vals = input_profile.init_vals # Let system settle with desired initial value for data sequence
        input_profile.prepend(val=warm_start_vals, length=warm_start_t // delta_t)

        save_name = ''.join((filename, '_output_clipped'))
        model_path = Path(__file__).parent / "../fmu/SingleWell_filtGas.fmu"
        model, y1, y2 = init_model(model_path, start_time, input_profile.end, 
                                Uk=warm_start_vals, warm_start_t=warm_start_t, 
                                warm_start=True)

        # ----- SIMULATING ----- #
        time = warm_start_t # As long as we've come after init
        warm_offset = warm_start_t // delta_t
        itr = 0
        while time < input_profile.end:
            inpt = input_profile[:, warm_offset + itr]
            y1k, y2k = simulate_singlewell_step(model, time, warm_start_t, delta_t, inpt)
            y1.append(y1k)
            y2.append(y2k)
            print(f'itr nr. {itr}\n')

            time += delta_t
            itr += 1


        # ----- PLOTTING ----- #

        fig, axs = plt.subplots(2, 2)
        plot_full=True
        if plot_full:
            # clip_length = 0
            clip_length = warm_start_t // delta_t
            t = clip_beginning(np.linspace(start=0, stop=input_profile.end, num=input_profile.end // delta_t), clip_length=clip_length)
            y1 = clip_beginning(y1, clip_length=clip_length)
            y2 = clip_beginning(y2, clip_length=clip_length)
            u1 = clip_beginning(input_profile[0, :], clip_length=clip_length)
            u2 = clip_beginning(input_profile[1, :], clip_length=clip_length)

            axs[0,0].plot(t, y1, label='gas rate', color='tab:orange')
            axs[0,0].legend(loc='best')
            axs[0,0].axvline(warm_start_t, color='tab:green')
            axs[1,0].plot(t, u1, label='choke')
            axs[1,0].legend(loc='best')
            axs[0,1].plot(t, y2, label='oil rate', color='tab:orange')
            axs[0,1].legend(loc='best')
            axs[0,1].axvline(warm_start_t, color='tab:green')
            axs[1,1].plot(t, u2, label='GL rate')
            axs[1,1].legend(loc='best')
        else:   
            t = np.linspace(start=warm_start_t, stop=input_profile.end - 1, num=(input_profile.end - warm_start_t) // delta_t)
            u1 = input_profile[0, warm_start_t // delta_t:]
            u2 = input_profile[1, warm_start_t // delta_t:]

            axs[0,0].plot(t, y1, label='gas rate', color='tab:orange')
            axs[0,0].legend(loc='best')
            axs[1,0].plot(t, u1, label='choke')
            axs[1,0].legend(loc='best')
            axs[0,1].plot(t, y2, label='oil rate', color='tab:orange')
            axs[0,1].legend(loc='best')
            axs[1,1].plot(t, u2, label='GL rate')
            axs[1,1].legend(loc='best')

        fig.suptitle('Step response')
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        choke_bounds = [0,100]
        GL_bounds = [0,10000]
        u1_normalized = normalize(u1, min=choke_bounds[0], max=choke_bounds[1])
        u2_normalized = normalize(u2, min=GL_bounds[0], max=GL_bounds[1])
        y1_normalized = normalize(y1)
        y2_normalized = normalize(y2)
        # ----- WRITE TO FILE ----- #
        header = ["t_" + str(int(timestamp)) for timestamp in t]
        csv_path = Path(__file__).parent / "data/steps_GL/csv" / ''.join((save_name, ".csv"))
        if not exists(csv_path):    # Safe to save; nothing can be overwritten
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(header)
                writer.writerow(u1_normalized)
                writer.writerow(u2_normalized)
                writer.writerow(y1_normalized)
                writer.writerow(y2_normalized)
            
            fig.savefig(Path(__file__).parent / "data/steps_GL/png" / ''.join((save_name, ".png")), bbox_inches='tight')
            fig.savefig(Path(__file__).parent / "data/steps_GL/eps" / ''.join((save_name, ".eps")), bbox_inches='tight')

        else:
            name = input("File already exists. Provide new name or \'y\' to overwrite ([enter] aborts. File-endings are automatic!): ")
            if name != '': # Do _not_ abort save
                if name != 'y': # Do _not_ overwrite
                    csv_path = Path(__file__).parent / "data/steps_GL/csv" / ''.join((name, '.csv'))
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)

                    writer.writerow(header)
                    writer.writerow(u1_normalized)
                    writer.writerow(u2_normalized)
                    writer.writerow(y1_normalized)
                    writer.writerow(y2_normalized)

                print(f"File written to \'{csv_path}\'")

                fig.savefig(Path(__file__).parent / "data/steps_GL/png" / ''.join((csv_path.stem, ".png")), bbox_inches='tight')
                fig.savefig(Path(__file__).parent / "data/steps_GL/eps" / ''.join((csv_path.stem, ".eps")), bbox_inches='tight')
            else:
                print("File was not saved.")
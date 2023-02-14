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
        for i in range(warm_start_t // delta_t): 
            model.set_real([3,4], [Uk[0, i], Uk[1, i]])
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
    # model.do_step(time + time_offset, delta_t) # Perform 1-step simulation of delta_t from time
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


if __name__ == '__main__':
    # for main_run_nr in range(0, 1):
    signal = 'step' # alternative: 'stair'
    step_type = 'choke' # alternative: 'GL'

    # ----- SETUP ----- #
    config_path = Path(__file__).parent / "../config/generate_data.yaml"
    with open(config_path, "r") as f:
        config = safe_load(f)
        if not config:
            Exception("Failed loading config file\n")

    start_time = config['start_time']
    warm_start_t = config['warm_start_t']
    step_time = warm_start_t + config['step_time'] # At what time the step should occur on the input
    # final_time = warm_start_t + config['final_time'] # For how long the total simulation will last
    delta_t = config['delta_t']
    choke = 0 # range: [0, 100]   [%]
    choke_step = 0
    choke_bounds = [0,100]
    GL = 0  # range: [0, 10000] [m^3/hr]
    GL_step = 2500
    GL_bounds = [0,10000] # 0 is logical, but 5000 is real boundary (in reality step between 0-5000)

    steps_path = Path(__file__).parent / "steps/steps40k.csv"
    input_profile = Timeseries(steps_path, delta_t=10)
    input_profile.prepend(val=[choke, GL], length=warm_start_t // delta_t)

    final_time = input_profile.length * delta_t

    # if signal == 'step':
    #     if step_type == 'choke':
    #         input_profile = np.array([step(start_time, final_time, step_time, delta_t=delta_t, 
    #                                         start_val=choke, step_val=choke_step),
    #                                     flatline(start_time, final_time, step_time, delta_t=delta_t, flat_val=GL)])
    #         save_name = signal + "_" + step_type + "_" + str(choke) + "_" + str(choke_step) + "_" + str(final_time - warm_start_t) + ".csv"
    #     elif step_type == 'GL':
    #         input_profile = np.array([flatline(start_time, final_time, step_time, delta_t=delta_t, start_val=choke, end_val=choke),
    #                                     step(start_time, final_time, step_time, delta_t=delta_t, 
    #                                         start_val=GL, end_val=GL_step)])
    #         save_name = signal + "_" + step_type + "_" + str(GL) + "_" + str(GL_step) + "_" + str(final_time - warm_start_t) + ".csv"

    # else:
    #     input_profile = np.array([staircase(init=choke, lb=choke_bounds[0], ub=choke_bounds[1], increment=2, interval=6, num=(final_time - start_time) // delta_t),
    #                               staircase(init=GL, lb=5000, ub=GL_bounds[1], increment=200, interval=6, num=(final_time - start_time) // delta_t)])
    #     save_name = signal + "_choke" + str(choke) + "_GL" + str(GL) + "_" + str(final_time - warm_start_t) + ".csv"
    
    save_name = "steps40k_output.csv"

    model_path = Path(__file__).parent / "../fmu/SingleWell_filtGas.fmu"
    model, y1_init, y2_init = init_model(model_path, start_time, final_time, 
                            Uk=input_profile, warm_start_t=warm_start_t, 
                            warm_start=True)


    # ----- SIMULATING ----- #
    time = warm_start_t # As long as we've come after init
    warm_offset = warm_start_t // delta_t
    itr = 0
    y1 = []
    y2 = []
    while time < final_time:
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
        t = np.linspace(start=0, stop=final_time - 1, num=final_time // delta_t)
        y1 = y1_init + y1
        y2 = y2_init + y2
        u1 = input_profile[0, :]
        u2 = input_profile[1, :]

        axs[0,0].plot(t, y1, label='gas rate', color='tab:orange')
        axs[0,0].legend()
        # axs[0,0].axvline(step_time, color='tab:red')
        axs[0,0].axvline(warm_start_t, color='tab:green')
        axs[1,0].plot(t, u1, label='choke')
        axs[1,0].legend()
        axs[0,1].plot(t, y2, label='oil rate', color='tab:orange')
        axs[0,1].legend()
        # axs[0,1].axvline(step_time, color='tab:red')
        axs[0,1].axvline(warm_start_t, color='tab:green')
        axs[1,1].plot(t, u2, label='GL rate')
        axs[1,1].legend()
    else:   
        t = np.linspace(start=warm_start_t, stop=final_time - 1, num=(final_time - warm_start_t) // delta_t)
        u1 = input_profile[0, warm_start_t // delta_t:]
        u2 = input_profile[1, warm_start_t // delta_t:]

        axs[0,0].plot(t, y1, label='gas rate', color='tab:orange')
        axs[0,0].legend()
        axs[1,0].plot(t, u1, label='choke')
        axs[1,0].legend()
        axs[0,1].plot(t, y2, label='oil rate', color='tab:orange')
        axs[0,1].legend()
        axs[1,1].plot(t, u2, label='GL rate')
        axs[1,1].legend()

    fig.suptitle('Step response')
    fig.tight_layout()

    plt.show()


    u1_normalized = normalize(u1, min=choke_bounds[0], max=choke_bounds[1])
    u2_normalized = normalize(u2, min=GL_bounds[0], max=GL_bounds[1])
    y1_normalized = normalize(y1)
    y2_normalized = normalize(y2)
    # ----- WRITE TO FILE ----- #
    header = ["t_" + str(int(timestamp)) for timestamp in t]
    # rel_path = "data/" + signal + "normalized_u1_" + str(choke) + "_u2_" + str(GL) + "_stairs_" + str(main_run_nr) + "_" + str(final_time - warm_start_t) + ".csv"
    csv_path = Path(__file__).parent / "data/" / save_name
    if not exists(csv_path):    # Safe to save; nothing can be overwritten
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)
            writer.writerow(u1_normalized)
            writer.writerow(u2_normalized)
            writer.writerow(y1_normalized)
            writer.writerow(y2_normalized)

    else:
        decision = input("File already exists. Provide new name or \'y\' to overwrite ([enter] aborts): ")
        if decision != '':
            if decision != 'y':
                csv_path = Path(__file__).parent / "data/" + decision
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(header)
                writer.writerow(u1_normalized)
                writer.writerow(u2_normalized)
                writer.writerow(y1_normalized)
                writer.writerow(y2_normalized)

            print(f"File written to \'{csv_path}\'")
        else:
            print("File was not saved.")
        
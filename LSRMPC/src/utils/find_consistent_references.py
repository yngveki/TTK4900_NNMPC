#!/usr/bin/env python3

from pathlib import Path
from itertools import product
import numpy as np

from simulate_fmu import init_model, simulate_singlewell_step
from saving import safe_save
from plotting import plot_IO

if __name__ == '__main__':
    # Set up
    fmu_path = Path(__file__).parent.parent.parent / ('fmu/fmu_endret_deadband.fmu')
    y_prev = np.zeros((2,1))
    u_prev_act = np.zeros((2,1))
    u_prev_meas = np.zeros((2,1))

    inputs_choke = [20,40,60,80,100]
    inputs_gl = [0,2000,4000,6000,8000,10000]
    n_inputs = len(inputs_choke) * len(inputs_gl)            
    inputs = product(inputs_choke, inputs_gl)

    delta_t = 10
    n_steps_per_input = 250
    final_t = n_inputs * delta_t * n_steps_per_input # 200 time steps per input-config
    time = 0
    warm_start_t = 2000

    full_choke_act = []
    full_gl_act = []
    full_choke_meas = []
    full_gl_meas = []
    full_gasrate = []
    full_oilrate = []
    
    # init model
    fmu, _, _ = init_model(fmu_path, start_time=time, final_time=final_t, delta_t=delta_t, warm_start_t=warm_start_t)
    time = warm_start_t

    # Let model settle for each configuration of input desired
    for i, inpt in enumerate(inputs):
        print(f'Iteration: {i+1}/{n_inputs}')
        print(f'Simulating for input: {inpt}')
        for _ in range(n_steps_per_input):
            U_next = inpt

            y_prev[0], y_prev[1], \
            u_prev_act[0], u_prev_act[1], \
            u_prev_meas[0], u_prev_meas[1] = simulate_singlewell_step(fmu, 
                                                                    time, 
                                                                    delta_t,
                                                                    warm_start_t, 
                                                                    U_next) # measurement from FMU, i.e. result from previous actuation

            full_choke_act.append(round(u_prev_act[0].item(), 5))
            full_gl_act.append(round(u_prev_act[1].item(), 5))
            full_choke_meas.append(round(u_prev_meas[0].item(), 5))
            full_gl_meas.append(round(u_prev_meas[1].item(), 5))
            full_gasrate.append(round(y_prev[0].item(), 5))
            full_oilrate.append(round(y_prev[1].item(), 5))

            time += delta_t
    
    save_dir = Path(__file__).parent.parent / ('tests/find_consistent_references/')
    # Plot results
    fig_dir = save_dir / ('choke20-100_gl0-10000')
    timeline = np.linspace(warm_start_t, time, num=(time-warm_start_t)//delta_t)
    fig = plot_IO(full_gasrate, full_oilrate, full_choke_act, full_gl_act, timeline, delta_t)
    safe_save(fig_dir, fig, 'fig', create_parent=True, errmsgstr='fig')
    # Save results?
    data_name = 'consistent_reachable_values.csv'
    data_dir = save_dir / data_name
    data = np.zeros((n_steps_per_input * n_inputs, 6))
    # data[0,:] = ['choke', 'gaslift', 'gasrate', 'oilrate', 'measured_choke', 'measured_gl']
    data[:,0] = full_choke_act
    data[:,1] = full_choke_meas
    data[:,2] = full_gl_act
    data[:,3] = full_gl_meas
    data[:,4] = full_gasrate
    data[:,5] = full_oilrate
    safe_save(data_dir, data, 'csv', create_parent=True, errmsgstr=f'{data_name}')
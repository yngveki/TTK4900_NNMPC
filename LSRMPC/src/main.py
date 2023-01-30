#!/usr/bin/env python3

from pathlib import Path

from MPC_oop import MPC
from utils.plotting import plot_LSRMPC
from utils.custom_timing import Timer

if __name__ == "__main__":
    config_path = Path(__file__).parent / "../config/mpc_config.yaml"
    S_paths = {'S11': '../data/S11_data_new.npy', 'S21': '../data/S12_data_new.npy',
               'S12': '../data/S21_data_new.npy', 'S22': '../data/S22_data_new.npy'}
    fmu_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"
    ref_path = Path(__file__).parent / "../config/refs0.csv"

    # Initialize the controller. Sets up all parameters and static matrices
    mpc = MPC(config_path, S_paths, ref_path)

    # Ensure FMU is in a defined state
    mpc.warm_start(fmu_path, warm_start_t=1000)

    timed_loop = True
    if timed_loop: stopwatch = Timer()
    # if timed_loop: stopwatch.start()
    run = 1
    total_runs = mpc.final_time // mpc.delta_t

    while mpc.time < mpc.final_time:
        if run % 10 == 0: print(f'Run #{run} / {total_runs}')

        # Update matrices and constraints that are time dependent
        if timed_loop: stopwatch.lap(silent=True)
        mpc.update_matrices()

        if timed_loop: stopwatch.lap(silent=True)
        mpc.update_OCP()

        # Solve OCP for this timestep
        if timed_loop: stopwatch.lap(silent=True)
        mpc.solve_OCP()

        # Simulate one step for the system
        if timed_loop: stopwatch.lap(silent=True)
        mpc.iterate_system()

        if timed_loop: stopwatch.total_time()
        run += 1

    # Save data, so plotting is possible later, without running the full simulation
    data_path = Path(__file__).parent / "../data/mpc_runs/"
    mpc.save_data(data_path)
    
    # Plot full simulation
    plot_LSRMPC(mpc)

# TODO: Make timing-suite decoratable
    
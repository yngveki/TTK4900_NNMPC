#!/usr/bin/env python3

from pathlib import Path

from MPC_oop import MPC
from plotting import plot_LSRMPC
from custom_timing import Timer

if __name__ == "__main__":
    config_path = Path(__file__).parent / "../config/mpc_config.yaml"
    S_paths = {'S11': '../data/S11_data_new.npy', 'S21': '../data/S12_data_new.npy',
               'S12': '../data/S21_data_new.npy', 'S22': '../data/S22_data_new.npy'}
    fmu_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"

    # Initialize the controller. Sets up all parameters and static matrices
    mpc = MPC(config_path, S_paths)

    # Ensure FMU is in a defined state
    mpc.warm_start(fmu_path, warm_start_t=1000)

    # Set input profile for this simulation
    steps = [(3000, (0, 10)),
             (5000, (0, 20)),
             (7000, (0, 10)),
             (9000, (1000, 10)),
             (11000, (1000, 0)),
             (13000, (0, -30)),
             (16000, (-5000, -30))] # TODO: Consider fetching this from file
    # TODO: mpc.step_input(steps)

    timed_loop = True
    if timed_loop: stopwatch = Timer()
    if timed_loop: stopwatch.start()
    run = 1
    total_runs = mpc.final_time // mpc.delta_t
    while mpc.time < mpc.final_time:
        print(f'Run #{run} / {total_runs}')
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

    # Plot full simulation
    plot_LSRMPC(mpc)
    
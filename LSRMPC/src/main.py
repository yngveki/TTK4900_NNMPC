#!/usr/bin/env python3

from pathlib import Path

from MPC_oop import MPC

if __name__ == "__main__":
    config_path = Path(__file__).parent / "../config/mpc_config.yaml"
    S_paths = {'S11': '../data/S11_data_new.npy', 'S21': '../data/S12_data_new.npy',
               'S12': '../data/S21_data_new.npy', 'S22': '../data/S22_data_new.npy'}
    fmu_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"

    # Initialize the controller. Sets up all parameters and static matrices
    mpc = MPC(config_path, S_paths)

    # Ensure FMU is in a defined state
    mpc.warm_start(fmu_path)

    # Set input profile for this simulation
    steps = [(3000, (0, 10)),
             (5000, (0, 20)),
             (7000, (0, 10)),
             (9000, (1000, 10)),
             (11000, (1000, 0)),
             (13000, (0, -30)),
             (16000, (-5000, -30))] # TODO: Consider fetching this from file
    mpc.step_input(steps)

    while True:
        # Update matrices and constraints that are time dependent
        mpc.update_matrices()
        mpc.update_OCP()

        # Solve OCP for this timestep
        mpc.solve_OCP()

        # Simulate one step for the system
        mpc.iterate_system()

        # Plot for this step
        # TODO

    # Plot full simulation
    # TODO
    
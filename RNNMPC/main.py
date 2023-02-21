#!/usr/bin/env python3

from pathlib import Path

from src.MPC import RNNMPC
from src.utils.custom_timing import Timer
from src.utils.plotting import plot_RNNMPC
if __name__ == "__main__":

    model_path = Path(__file__).parent / "models/model_prosjektoppgave_6.pt"
    fmu_path = Path(__file__).parent / "fmu/fmu_endret_deadband.fmu"
    ref_path = Path(__file__).parent / "config/refs/refs0.csv"
    mpc_config_path = Path(__file__).parent / "config/mpc_config.yaml"
    nn_config_path = Path(__file__).parent / "config/nn_config.yaml"
    fig_save_path = Path(__file__).parent / "figs/mpc/test_fig.png"

    # Initialize the controller
    mpc = RNNMPC(nn_path=model_path,
                 mpc_config_path=mpc_config_path,
                 nn_config_path=nn_config_path,
                 ref_path=ref_path)

    # Ensure FMU is in a defined state
    mpc.warm_start(fmu_path)
    timed_loop = True
    if timed_loop: stopwatch = Timer()
    if timed_loop: stopwatch.start()
    run = 1
    total_runs = (mpc.final_t - mpc.warm_start_t) // mpc.delta_t

    while mpc.t < mpc.final_t:
        print(f'Run #{run} / {total_runs}')

        if timed_loop: stopwatch.lap(silent=False)
        mpc.update_OCP()

        # Solve OCP for this timestep
        if timed_loop: stopwatch.lap(silent=False)
        mpc.solve_OCP()

        # Simulate one step for the system
        if timed_loop: stopwatch.lap(silent=False)
        mpc.iterate_system()

        if timed_loop: stopwatch.total_time()
        run += 1
    
    mpc.merge_sim_data()
    mpc.save_data(data_path=Path(__file__).parent / "mpc_data/")

    plot_RNNMPC(mpc=mpc, save_path=fig_save_path)
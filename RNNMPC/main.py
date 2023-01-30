#!/usr/bin/env python3

from pathlib import Path

from src.MPC import RNNMPC
from src.utils.plotting import plot_LSRMPC
from src.utils.custom_timing import Timer

if __name__ == "__main__":

    fmu_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"
    mpc = RNNMPC()

    # Ensure FMU is in a defined state
    mpc.warm_start(fmu_path, warm_start_t=1000)

    timed_loop = True
    if timed_loop: stopwatch = Timer()
    if timed_loop: stopwatch.start()
    run = 1
    total_runs = mpc.final_time // mpc.delta_t

    while mpc.time < mpc.final_time:
        if run % 10 == 0: print(f'Run #{run} / {total_runs}')

        if timed_loop: stopwatch.lap(silent=True)
        mpc.update_OCP()

        if timed_loop: stopwatch.lap(silent=True)
        mpc.solve_OCP()

        if timed_loop: stopwatch.lap(silent=True)
        mpc.iterate_system()

        if timed_loop: stopwatch.total_time()
        run += 1
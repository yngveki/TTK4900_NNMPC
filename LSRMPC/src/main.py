#!/usr/bin/env python3

from pathlib import Path
from yaml import safe_load
from itertools import product

from MPC import MPC
from utils.plotting import plot_LSRMPC
from utils.custom_timing import Timer
from utils.saving import safe_save

# -- FUNCS -- #
def grid_search_params(config_path, searchable):
    '''
    Parses a config file with lists of candidate values for all parameters 
    into a list of parameter sets, grid search-style

    Args:
        :param config_path: path from which the config-values should be read
        :param searchable: an iterable of strings describing which config-values should be allowed to search over
                           (e.g. hard constraints are absolute; not to be tuned) 
    '''
    
    def parse_to_dict(values, keys):
        '''
        Parses an individual combination of parameter values back to dictionary format
        
        Note that \'m\' is implemented this way because mu and my must always be alike,
        so they cannot be implemented as two distinct lists in the config-file
        '''
        d = {}
        for val, key in zip(values, keys):
            if key in ['PLUGIN_OPTIONS', 'SOLVER_OPTIONS']:
                continue

            d[key] = val

        return d
    
    with open(config_path, "r") as f:
        params = safe_load(f)

    all_param_sets = []

    # Set up grid
    searchable_params = {}
    for key, val in params.items():
        if key in searchable:
            searchable_params[key] = val
            
    grid = product(*searchable_params.values())
    for point in grid:
        point_on_grid = parse_to_dict(point, searchable_params.keys()) # So far incomplete
        for key, val in params.items():
            if key not in point_on_grid:
                point_on_grid.update({key: val})

        all_param_sets.append(point_on_grid)

    return all_param_sets

# -- SCRIPT BEGINS -- #

GRID = True

if GRID == True:
    mpc_config_path = Path(__file__).parent / '../config/mpc_config_grid.yaml'
    sets = grid_search_params(mpc_config_path, ('R_bar','Q_bar'))

    to_pop = []
    for i, s in enumerate(sets):
        if s['Hu'] > s['Hp']:
            to_pop.append(i)
    to_pop.reverse()
    for i in to_pop:
        sets.pop(i)

else:
    mpc_config_path = Path(__file__).parent / '../config/mpc_config.yaml'
    with open(mpc_config_path, "r") as f:
        params = safe_load(f)
    sets = [params]

for i, params in enumerate(sets):
    try:
        S_paths = {'S11': '../LSRmodel/S11_data_new.npy', 'S21': '../LSRmodel/S12_data_new.npy',
                'S12': '../LSRmodel/S21_data_new.npy', 'S22': '../LSRmodel/S22_data_new.npy'}
        fmu_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"
        ref_path = Path(__file__).parent / "../config/ref_const_1.csv"
        parent_dir = Path(__file__).parent / '../mpc_tunings'

        config_name = 'grid3_v2_' + str(i)

        # Initialize the controller. Sets up all parameters and static matrices
        mpc = MPC(params, S_paths, ref_path)

        # Ensure FMU is in a defined state
        mpc.warm_start(fmu_path, warm_start_input=[90,0])

        timed_loop = True
        if timed_loop: stopwatch = Timer()
        if timed_loop: stopwatch.start()
        run = 1
        total_runs = mpc.final_time // mpc.delta_t
        
        # Timekeeping (for data for results-section)
        t_update_matrices = []
        t_update_OCP = []
        t_solve_OCP = []
        t_iterate_system = []
        t_full_loop = []

        # Logging objective function values in order to measure qualitative performance of any given tuning
        objective_vals = []

        while mpc.time < mpc.final_time:
            if run % 10 == 0: print(f'Run #{run} / {total_runs}')

            # Update matrices and constraints that are time dependent
            if timed_loop: stopwatch.lap(silent=True)
            mpc.update_matrices()

            if timed_loop: t_update_matrices.append(stopwatch.lap(silent=True, ret=True))
            mpc.update_OCP()

            # Solve OCP for this timestep
            if timed_loop: t_update_OCP.append(stopwatch.lap(silent=True, ret=True))
            objective_vals.append(mpc.solve_OCP())

            # Simulate one step for the system
            if timed_loop: t_solve_OCP.append(stopwatch.lap(silent=True, ret=True))
            mpc.iterate_system()

            if timed_loop: t_iterate_system.append(stopwatch.lap(silent=True, ret=True))
            run += 1
            if timed_loop: t_full_loop.append(stopwatch.total_time())

        # Plot full simulation
        fig = plot_LSRMPC(mpc)

        # -- SAVING MPC TUNING ALONG WITH RESULTS -- #
        # Appending files used for this specific run
        params['FILES'] = {'fmu': fmu_path.__str__(),
                            'refs': ref_path.__str__()}
        save_dir = parent_dir / config_name

        # -- SAVING CONFIG -- #
        safe_save(save_dir / (config_name + '.yaml'), params, 'yaml', create_parent=True, errmsgstr='config')

        # -- SAVING MPC-DATA -- #
        safe_save(save_dir / 'data/t.npy', mpc.t, 'npy', create_parent=True, errmsgstr='t.npy')
        safe_save(save_dir / 'data/oil_rate_per_hr_vec.npy', mpc.oil_rate_per_hr_vec, 'npy', create_parent=True, errmsgstr='oil_rate_per_hr_vec.npy')
        safe_save(save_dir / 'data/oil_rate_ref_vec.npy', mpc.oil_rate_ref_vec, 'npy', create_parent=True, errmsgstr='oil_rate_ref_vec.npy')
        safe_save(save_dir / 'data/gas_rate_per_hr_vec.npy', mpc.gas_rate_per_hr_vec, 'npy', create_parent=True, errmsgstr='gas_rate_per_hr_vec.npy')
        safe_save(save_dir / 'data/gas_rate_ref_vec.npy', mpc.gas_rate_ref_vec, 'npy', create_parent=True, errmsgstr='gas_rate_ref_vec.npy')
        safe_save(save_dir / 'data/choke_input.npy', mpc.choke_input, 'npy', create_parent=True, errmsgstr='choke_input.npy')
        safe_save(save_dir / 'data/gas_lift_input.npy', mpc.gas_lift_input, 'npy', create_parent=True, errmsgstr='gas_lift_input.npy')
        safe_save(save_dir / 'data/choke_actual.npy', mpc.choke_actual, 'npy', create_parent=True, errmsgstr='choke_actual.npy')
        safe_save(save_dir / 'data/gas_lift_actual.npy', mpc.gas_lift_actual, 'npy', create_parent=True, errmsgstr='gas_lift_actual.npy')
        safe_save(save_dir / 'data/bias_gas.npy', mpc.bias_gas, 'npy', create_parent=True, errmsgstr='bias_gas.npy')
        safe_save(save_dir / 'data/bias_oil.npy', mpc.bias_oil, 'npy', create_parent=True, errmsgstr='bias_oil.npy')
        safe_save(save_dir / 'data/objective_function.npy', objective_vals, 'npy', create_parent=True, errmsgstr='objective_function.npy')

        # -- SAVING TIMEKEEPING -- #
        safe_save(save_dir / ('t/t_update_matrices.npy'), t_update_matrices, 'npy', create_parent=True, errmsgstr='t_update_matrices.npy')
        safe_save(save_dir / ('t/t_update_OCP.npy'), t_update_OCP, 'npy', create_parent=True, errmsgstr='t_update_OCP.npy')
        safe_save(save_dir / ('t/t_solve_OCP.npy'), t_solve_OCP, 'npy', create_parent=True, errmsgstr='t_solve_OCP.npy')
        safe_save(save_dir / ('t/t_iterate_system.npy'), t_iterate_system, 'npy', create_parent=True, errmsgstr='t_iterate_system.npy')
        safe_save(save_dir / ('t/t_full_loop.npy'), t_full_loop, 'npy', create_parent=True, errmsgstr='t_full_loop.npy')

        # -- SAVING FIGS -- #
        fig_dir = save_dir / ('figs/' + str(params['final_time'] // params['delta_t']) + '_steps.png')
        safe_save(fig_dir,
                    fig,
                    'fig',
                    create_parent=True,
                    errmsgstr=(str(params['final_time'] // params['delta_t']) + '_steps'))
        
    
    except Exception as error: # To safeguard against exiting on solution failure
        print(f'An error occurred: {error}')

        parent_dir = Path(__file__).parent / '../mpc_tunings'
        save_dir = parent_dir / config_name
        config_save_path = save_dir / (config_name + '_error.yaml')
        safe_save(config_save_path, params, 'yaml', create_parent=True, errmsgstr='config')
        continue
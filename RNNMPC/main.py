#!/usr/bin/env python3

from pathlib import Path
from yaml import safe_load
from itertools import product

from src.MPC import RNNMPC
from src.utils.custom_timing import Timer
from src.utils.plotting import plot_RNNMPC
from src.utils.saving import safe_save

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
        """
        Parses an individual combination of parameter values back to dictionary format
        
        Note that \'m\' is implemented this way because mu and my must always be alike,
        so they cannot be implemented as two distinct lists in the config-file
        """
        d = {}
        for val, key in zip(values, keys):
            if key in ['PLUGIN_OPTIONS', 'SOLVER_OPTIONS']:
                continue

            d[key] = val

        d['Hp'] = d['H']
        d['Hu'] = d['H']
        del d['H']
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
    mpc_config_path = Path(__file__).parent / 'config/mpc_config_grid.yaml'
    sets = grid_search_params(mpc_config_path, ('H', 'Q', 'R'))
else:
    mpc_config_path = Path(__file__).parent / 'config/mpc_config.yaml'
    with open(mpc_config_path, "r") as f:
        params = safe_load(f)
    sets = [params]

for i, params in enumerate(sets):
    try:
    # if __name__ == '__main__':
        # BEST MODEL
        # model_name = 'model_grid_second_run_0'
        if i <= -1:
            continue

        # FASTER MODEL - still very decent (little to no offset?)
        model_name = 'light_weight_0'
        model_path = Path(__file__).parent / ('models/' + model_name + '/' + model_name + '.pt')
        fmu_path = Path(__file__).parent / 'fmu/fmu_endret_deadband.fmu'
        ref_path = Path(__file__).parent / 'config/refs/ref_const_1.csv'
        nn_config_path = Path(__file__).parent / ('models/' + model_name + '/' + model_name + '.yaml')
        # fig_save_path = Path(__file__).parent / 'figs/mpc/test_fig.png'
        parent_dir = Path(__file__).parent / 'mpc_tunings'

        config_name = model_name + '_gridnr_5_' + str(i) # For saving the config along with results

        # Initialize the controller
        mpc = RNNMPC(nn_path=model_path,
                    nn_config_path=nn_config_path,
                    ref_path=ref_path,
                    mpc_configs=params)

        # Ensure FMU is in a defined state
        warm_start_input = [90,5000]
        mpc.warm_start(fmu_path, warm_start_input)
        timed_loop = True
        if timed_loop: stopwatch = Timer()
        if timed_loop: stopwatch.start()
        run = 1
        total_runs = (mpc.final_t - mpc.warm_start_t) // mpc.delta_t

        # Timekeeping (for data for results-section)
        t_update_OCP = []
        t_solve_OCP = []
        t_iterate_system = []
        t_full_loop = []

        while mpc.t < mpc.final_t:
            print(f'Run #{run} / {total_runs}')

            if timed_loop: stopwatch.lap(silent=False)
            mpc.update_OCP()

            # Solve OCP for this timestep
            if timed_loop: t_update_OCP.append(stopwatch.lap(silent=False, ret=True))
            mpc.solve_OCP()

            # Simulate one step for the system
            if timed_loop: t_solve_OCP.append(stopwatch.lap(silent=False, ret=True))
            mpc.iterate_system()

            if timed_loop: t_iterate_system.append(stopwatch.lap(silent=False, ret=True))
            run += 1
            if timed_loop: t_full_loop.append(stopwatch.total_time(ret=True))

        # Return fig for saving purposes
        fig = plot_RNNMPC(mpc=mpc, pause=True, plot_bias=True)

        # -- SAVING MPC TUNING ALONG WITH RESULTS -- #
        # Appending files used for this specific run
        params['FILES'] = {'model': model_path.__str__(),
                        'fmu': fmu_path.__str__(),
                        'refs': ref_path.__str__()}
        save_dir = parent_dir / config_name

        # -- SAVING CONFIG -- #
        safe_save(save_dir / (config_name + '.yaml'), params, 'yaml', create_parent=True, errmsgstr='config')
        
        # -- SAVING MPC-DATA -- #
        safe_save(save_dir / 'data/t.npy', mpc.full_t, 'npy', create_parent=True, errmsgstr='t.npy')
        safe_save(save_dir / 'data/gas_rate.npy', mpc.simulated_y['gas rate'], 'npy', create_parent=True, errmsgstr='gas_rate')
        safe_save(save_dir / 'data/oil_rate.npy', mpc.simulated_y['oil rate'], 'npy', create_parent=True, errmsgstr='oil_rate.npy')
        safe_save(save_dir / 'data/choke.npy', mpc.simulated_u['choke'], 'npy', create_parent=True, errmsgstr='choke.npy')
        safe_save(save_dir / 'data/gas_lift.npy', mpc.simulated_u['gas lift'], 'npy', create_parent=True, errmsgstr='gas_lift.npy')
        safe_save(save_dir / 'data/gas_rate_ref.npy', mpc.full_refs['gas rate'], 'npy', create_parent=True, errmsgstr='gas_rate_ref.npy')
        safe_save(save_dir / 'data/oil_rate_ref.npy', mpc.full_refs['oil rate'], 'npy', create_parent=True, errmsgstr='oil_rate_ref.npy')
        
        # -- SAVING TIMEKEEPING -- #
        safe_save(save_dir / ('t/t_update_OCP.npy'), t_update_OCP, 'npy', create_parent=True, errmsgstr='t_update_OCP.npy')
        safe_save(save_dir / ('t/t_solve_OCP.npy'), t_solve_OCP, 'npy', create_parent=True, errmsgstr='t_solve_OCP.npy')
        safe_save(save_dir / ('t/t_iterate_system.npy'), t_iterate_system, 'npy', create_parent=True, errmsgstr='t_iterate_system.npy')
        safe_save(save_dir / ('t/t_full_loop.npy'), t_full_loop, 'npy', create_parent=True, errmsgstr='t_full_loop.npy')

        # -- SAVING FIGS -- #
        fig_dir = save_dir / ('figs/' + str(params['final_t'] // params['delta_t']) + '_steps')
        safe_save(fig_dir,
                    fig,
                    'fig',
                    create_parent=True,
                    errmsgstr=(str(params['final_t'] // params['delta_t']) + '_steps'))

    except: # To safeguard against exiting on solution failure
        parent_dir = Path(__file__).parent / 'mpc_tunings'
        save_dir = parent_dir / config_name
        config_save_path = save_dir / (config_name + '.yaml')
        safe_save(config_save_path, params, 'yaml', create_parent=True, errmsgstr='config')
        continue
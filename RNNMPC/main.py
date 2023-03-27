#!/usr/bin/env python3

from pathlib import Path
from os.path import exists
from os import makedirs
from yaml import safe_load, dump
from itertools import product

from src.MPC import RNNMPC
from src.utils.custom_timing import Timer
from src.utils.plotting import plot_RNNMPC

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

GRID = False

if GRID == True:
    mpc_config_path = Path(__file__).parent / 'config/mpc_config_grid.yaml'
    sets = grid_search_params(mpc_config_path, ('H', 'Q', 'R', 'rho'))
else:
    mpc_config_path = Path(__file__).parent / 'config/mpc_config.yaml'
    with open(mpc_config_path, "r") as f:
        params = safe_load(f)
    sets = [params]


for i, params in enumerate(sets):
    if __name__ == '__main__':

        # BEST MODEL
        # model_name = 'model_grid_second_run_8'

        # FASTER MODEL - still very decent (little to no offset?)
        model_name = 'model_grid_0'
        model_path = Path(__file__).parent / ('models/' + model_name + '/' + model_name + '.pt')
        fmu_path = Path(__file__).parent / 'fmu/fmu_endret_deadband.fmu'
        ref_path = Path(__file__).parent / 'config/refs/refs0.csv'
        mpc_config_path = Path(__file__).parent / 'config/mpc_config.yaml'
        nn_config_path = Path(__file__).parent / ('models/' + model_name + '/' + model_name + '.yaml')
        fig_save_path = Path(__file__).parent / 'figs/mpc/test_fig.png'

        config_name = 'grid_search_' + str(i) # For saving the config along with results

        # Initialize the controller
        mpc = RNNMPC(nn_path=model_path,
                    nn_config_path=nn_config_path,
                    ref_path=ref_path,
                    mpc_configs=params)

        # Ensure FMU is in a defined state
        warm_start_input = [50,0]
        mpc.warm_start(fmu_path, warm_start_input)
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

        # Return fig for saving purposes
        fig = plot_RNNMPC(mpc=mpc, pause=True, plot_bias=True)

        # -- SAVING MPC TUNING ALONG WITH RESULTS -- #
        # Appending files used for this specific run
        with open(mpc_config_path, 'r') as f:
            config = safe_load(f)

        config['FILES'] = {'model': model_path.__str__(),
                        'fmu': fmu_path.__str__(),
                        'refs': ref_path.__str__()}
        
        # Setting up directory for results
        parent_dir = Path(__file__).parent / 'mpc_tunings'
        save_dir = parent_dir / config_name
        if not exists(save_dir):
            makedirs(save_dir)

        config_save_path = save_dir / (config_name + '.yaml')

        if exists(config_save_path):
            name = input("Config with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if name != '': # _not_ aborting
                if name != 'y': # do _not_ want to override
                    config_save_path = save_dir / (name + '.yaml')

            else:
                config_save_path = None
        
        if config_save_path is not None:
            # Save config
            with open(config_save_path, 'w', encoding = 'utf-8') as yaml_file:
                yaml_file.write(dump(config, default_flow_style = False, allow_unicode = True, encoding = None))
                mpc.save_data(config_save_path.parent)
        else:
            print('Config was not saved')

        # -- SAVING FIGS -- #    
        save_dir /= 'figs'
        if not exists(save_dir):
            makedirs(save_dir)
        save_path = save_dir / (str(config['final_t'] // config['delta_t']) + '_steps')

        if      exists(save_path.parent / (save_path.stem + '.png')) \
            or  exists(save_path.parent / (save_path.stem + '.eps')):
            name = input("Figure(s) with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if name != '': # _not_ aborting
                if name != 'y': # do _not_ want to override
                    save_path = save_path.parent / name

            else:
                save_path = None
            
        if save_path is not None:
            for suffix in ['.png', '.eps']:
                save_path = save_path.parent / (save_path.stem + suffix)
                fig.savefig(save_path, bbox_inches='tight')
        else:
            print("Figures were not saved.")
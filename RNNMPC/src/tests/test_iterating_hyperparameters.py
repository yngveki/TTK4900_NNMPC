from pathlib import Path
from yaml import safe_load
from itertools import product

def grid_search_params(config_path):
    """
    Parses a config file with lists of candidate values for all parameters 
    into a list of parameter sets, grid search-style
    """
    
    def parse_to_dict(values, keys):
        """
        Parses an individual combination of parameter values back to dictionary format
        
        Note that \'m\' is implemented this way because mu and my must always be alike,
        so they cannot be implemented as two distinct lists in the config-file
        """
        d = {}
        for val, key in zip(values, keys):
            d[key] = val
        d['mu'] = d['m']
        d['my'] = d['m']
        del d['m']
        return d
    
    with open(config_path, "r") as f:
        params = safe_load(f)

    all_param_sets = []
    
    # Need to unwrap layers of dictionaries
    
    
    # Set up grid
    grid = product(*params.values())
    for point in grid:
        all_param_sets.append(parse_to_dict(point, params.keys()))

    return all_param_sets

if __name__ == '__main__':
    hyperparameter_name = '../../config/nn_config_grid.yaml'
    hyperparameter_path = Path(__file__).parent / hyperparameter_name

    sets = grid_search_params(hyperparameter_path)
    for set in sets:
        print(set)

    print('finished')
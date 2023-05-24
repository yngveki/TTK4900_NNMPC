#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from yaml import safe_load
from os.path import exists

if __name__ == '__main__':
    parent_dir = Path(__file__).parent / ('../../mpc_tunings')
    tuning_name = 'light_weight_0_gridnr_5'
    num_tunings = 8
    MSEs_gasrate = [0] * num_tunings
    MSEs_oilrate = [0] * num_tunings

    for i in range(num_tunings):
        # Retrieve simulation time so data may later be truncated
        config_path = parent_dir / (tuning_name + '_' + str(i) + '/' + tuning_name + '_' + str(i) + '.yaml')
        with open(config_path, "r") as f:
            config = safe_load(f)
        num_sim_steps = config['final_t'] // config['delta_t']

        # Retrieve data. Truncate (cut off warm start period)
        gasrate_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/gas_rate.npy')
        gasrateref_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/gas_rate_ref.npy')
        oilrate_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/oil_rate.npy')
        oilrateref_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/oil_rate_ref.npy')

        # This run succeeded
        if exists(gasrate_dir) and exists(gasrate_dir) and exists(gasrate_dir) and exists(gasrate_dir):
            # Denormalize normalized fetched values
            gasrate = np.multiply(np.load(gasrate_dir)[-num_sim_steps:], 18539)
            gasrateref = np.multiply(np.load(gasrateref_dir)[-num_sim_steps:], 18539)
            oilrate = np.multiply(np.load(oilrate_dir)[-num_sim_steps:], 349)
            oilrateref = np.multiply(np.load(oilrateref_dir)[-num_sim_steps:], 349)

            # Calculate MSE and store together with index. Entry is later sort-able
            MSEs_gasrate[i] = {'index':i, 'MSE': np.mean(np.square(gasrate - gasrateref))}
            MSEs_oilrate[i] = {'index':i, 'MSE': np.mean(np.square(oilrate - oilrateref))}

        # This run failed
        else:
            MSEs_gasrate[i] = {'index':i, 'MSE': np.inf, 'errmsg': 'this config failed'}
            MSEs_oilrate[i] = {'index':i, 'MSE': np.inf, 'errmsg': 'this config failed'}


    MSEs_gasrate = sorted(MSEs_gasrate, key=lambda x: x['MSE'])
    MSEs_oilrate = sorted(MSEs_oilrate, key=lambda x: x['MSE'])

    print('Top-down sorted MSEs for gas rate')
    for g in MSEs_gasrate:
        MSE_g = g['MSE']
        MSE_i = g['index']
        print(f'Grid search \'{tuning_name}\', index {MSE_i:<4}. MSE gas rate: {MSE_g:<10.2f}.')

    print('Top-down sorted MSEs for oil rate')
    for o in MSEs_oilrate:
        MSE_o = o['MSE']
        MSE_i = o['index']
        print(f'Grid search \'{tuning_name}\', index {MSE_i:<4}. MSE gas rate: {MSE_o:<10.2f}.')
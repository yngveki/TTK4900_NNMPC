#!/usr/bin/env python3

import numpy as np
from pathlib import Path

if __name__ == '__main__':
    parent_dir = Path(__file__).parent / ('../../mpc_tunings')
    tuning_name = 'grid3_v2'
    num_tunings = 8
    MSEs_gasrate = [0] * num_tunings
    MSEs_oilrate = [0] * num_tunings

    for i in range(num_tunings):
        gasrate_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/gas_rate_per_hr_vec.npy')
        gasrateref_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/gas_rate_ref_vec.npy')
        oilrate_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/oil_rate_per_hr_vec.npy')
        oilrateref_dir = parent_dir / (tuning_name + '_' + str(i) + '/data/oil_rate_ref_vec.npy')

        gasrate = np.load(gasrate_dir)
        gasrateref = np.load(gasrateref_dir)
        oilrate = np.load(oilrate_dir)
        oilrateref = np.load(oilrateref_dir)

        # Calculate MSE and store together with index. Entry is later sort-able
        MSEs_gasrate[i] = {'index':i, 'MSE': np.mean(np.square(gasrate - gasrateref))}
        
        MSEs_oilrate[i] = {'index':i, 'MSE': np.mean(np.square(oilrate - oilrateref))}

    MSEs_gasrate = sorted(MSEs_gasrate, key=lambda x: x['MSE'])
    MSEs_oilrate = sorted(MSEs_oilrate, key=lambda x: x['MSE'])

    print('Top-down sorted MSEs for gas rate')
    for g in MSEs_gasrate:
        MSE_g = g['MSE']
        MSE_i = g['index']
        print(f'Grid search \'{tuning_name}\', index \'{MSE_i}\'. MSE gas rate: {MSE_g:<10.2f}.')

    print('Top-down sorted MSEs for oil rate')
    for o in MSEs_oilrate:
        MSE_o = o['MSE']
        MSE_i = o['index']
        print(f'Grid search \'{tuning_name}\', index \'{MSE_i}\'. MSE gas rate: {MSE_o:<10.2f}.')
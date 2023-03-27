from pathlib import Path
from os.path import exists
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from dataclasses import dataclass

def plot_RNNMPC(mpc=None, warm_start_cutoff=True, pause=True, plot_bias=False):
    """
    Plots data collected from a simulation of control by means of an RNNMPC
    
    See LSRMPC/src/plotting.py for """
    if mpc is None:
        """The idea is to take in some path to the data, verify that
        there is indeed data there, then read that data into variables that
        may be used for plotting.
        
        This implies that the data fetched directly from the mpc-object have
        to be confined also to the same variables that the data within this
        if-clause use, e.g. choke, gas_lift, gas_rate, oil_rate and t"""
        return NotImplementedError
    
    num = len(mpc.simulated_u['choke'])
    t = np.linspace(0, num * mpc.delta_t, num=num)
    cutoff = 0
    if warm_start_cutoff:
        cutoff = mpc.warm_start_t // mpc.delta_t

    fig, axes = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f'RNNMPC simulated {num} steps of {mpc.delta_t} [s] each. Total time is {mpc.final_t}', fontsize=23)

    # Plotting ground truth and predicted gas rates
    axes[0,0].set_title('Measured gas rate v. reference gas rate', fontsize=20)
    axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
    axes[0,0].plot(t[cutoff:], mpc.simulated_y['gas rate'][cutoff:], '-', label='true gas rate', color='tab:orange')
    axes[0,0].plot(t[-len(mpc.full_refs['gas rate']):], mpc.full_refs['gas rate'], '--', label='reference gas rate', color='tab:red')
    if not warm_start_cutoff: axes[0,0].axvline(mpc.warm_start_t, color='tab:green')
    axes[0,0].legend(loc='best', prop={'size': 15})

    if plot_bias: 
        twin00 = axes[0,0].twinx()
        twin00.plot(t[-len(mpc.full_refs['gas rate']):], mpc.bias['gas rate'][1:], label='bias gas rate', color='tab:olive')
        twin00.set_ylabel('gas rate bias (yk - yk_hat) [m^3/h]', color='tab:olive')
        twin00.tick_params(axis='y', color='tab:olive', labelcolor='tab:olive')
        twin00.spines['right'].set_color('tab:olive')
        max_lim = max(mpc.bias['gas rate']) + (2.5 * abs(max(mpc.bias['gas rate'])))
        min_lim = min(mpc.bias['gas rate']) - (0.5 * abs(min(mpc.bias['gas rate'])))
        twin00.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

    # Plotting ground truth and predicted oil rates
    axes[0,1].set_title('Measured oil rate v. reference oil rate', fontsize=20)
    axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
    axes[0,1].plot(t[cutoff:], mpc.simulated_y['oil rate'][cutoff:], label='true oil rate', color='tab:orange')
    axes[0,1].plot(t[-len(mpc.full_refs['oil rate']):], mpc.full_refs['oil rate'], '--', label='reference oil rate', color='tab:red')
    if not warm_start_cutoff: axes[0,1].axvline(mpc.warm_start_t, color='tab:green')
    axes[0,1].legend(loc='best', prop={'size': 15})

    if plot_bias: 
        twin01 = axes[0,1].twinx()
        twin01.plot(t[-len(mpc.full_refs['oil rate']):], mpc.bias['oil rate'][1:], label='bias oil rate', color='tab:olive')
        twin01.set_ylabel('oil rate bias (yk - yk_hat) [m^3/h]', color='tab:olive')
        twin01.tick_params(axis='y', color='tab:olive', labelcolor='tab:olive')
        twin01.spines['right'].set_color('tab:olive')
        max_lim = max(mpc.bias['oil rate']) + (2.5 * abs(max(mpc.bias['oil rate'])))
        min_lim = min(mpc.bias['oil rate']) - (0.5 * abs(min(mpc.bias['oil rate'])))
        twin01.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

    # Plotting history of choke input
    axes[1,0].set_title('Input: choke opening', fontsize=20)
    axes[1,0].set_xlabel('time [s]', fontsize=15)
    axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
    axes[1,0].plot(t[cutoff:], mpc.simulated_u['choke'][cutoff:], label='choke', color='blue')
    if not warm_start_cutoff: axes[1,0].axvline(mpc.warm_start_t, color='tab:green')
    axes[1,0].legend(loc='best', prop={'size': 15})

    # Plotting history of gas lift rate input
    axes[1,1].set_title('Input: gas lift rate', fontsize=20)
    axes[1,1].set_xlabel('time [s]', fontsize=15)
    axes[1,1].set_ylabel('percent opening [m^3/h]', fontsize=15)
    axes[1,1].plot(t[cutoff:], mpc.simulated_u['gas lift'][cutoff:], label='gas lift rate', color='blue')
    if not warm_start_cutoff: axes[1,1].axvline(mpc.warm_start_t, color='tab:green')
    axes[1,1].legend(loc='best', prop={'size': 15})

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    if pause:
        plt.pause(10)
        plt.close()

    # Saving outsourced to main by returning fig    
    return fig

# @dataclass
# class PlotElement():
#     """A dataclass containing everything needed to plot a graph onto an axis"""
#     def __init__(self, ):
#         ...
        
#         axs[0].set_title(outputs['title'], fontsize=20)
#         axs[0].set_ylabel(outputs['gas rate']['ylabel'], fontsize=15, color=outputs['gas rate']['color'])
#         axs[0].plot(t[4:], outputs['gas rate']['future'], '-', label=outputs['gas rate']['label'], color=outputs['gas rate']['color'])
#         axs[0].legend(loc='upper right', prop={'size': 15})
def plot_MPC_step(t, inputs, outputs, k=0, pause=True, pause_time=5, save_path=None):
    """
    Plots inputs and outputs from an arbitrary MPC step, with potentially past data, in addition to the current and future
    
    args:
        :param t: array of timestamps covering all timesteps for both inputs and outputs (which may be of inequal lengths)
        :param inputs: dictionary containing dictionaries of values for each input, describing past, current and future input values.
        :param outputs: dictionary containing dictionaries of values for each output, describing past, current and future input values.
        :param k: the 0-indexed index for the current timestep. If k > 0, any values < k are past, and all values > k are future.
    """

    assert k >= 0, 'current timestep must be a positive semi-definite index'
    # assert len(inputs) == 2, 'Expected exactly 2 inputs: choke and gas lift'
    # assert len(outputs) == 2, 'Expected exactly 2 outputs: gas rate and oil rate'

    fig, axs = plt.subplots(2, sharex=True)

    # gas rate
    axs[0].set_title(outputs['title'], fontsize=20)
    axs[0].set_ylabel(outputs['gas rate']['ylabel'], fontsize=15, color=outputs['gas rate']['color'])
    axs[0].plot(t[4:], outputs['gas rate']['future'], '-', label=outputs['gas rate']['label'], color=outputs['gas rate']['color'])
    axs[0].legend(loc='upper right', prop={'size': 15})

    # oil rate
    axs0_twin = axs[0].twinx()
    axs0_twin.set_ylabel(outputs['oil rate']['ylabel'], fontsize=15, color=outputs['oil rate']['color'])
    axs0_twin.plot(t[4:], outputs['oil rate']['future'], '-', label=outputs['oil rate']['label'], color=outputs['oil rate']['color'])
    axs0_twin.legend(loc='upper right', prop={'size': 15})
    
    # choke
    axs[1].set_title(inputs['title'], fontsize=20)
    axs[1].set_ylabel(inputs['choke']['ylabel'], fontsize=15, color=inputs['choke']['color'])
    axs[1].plot(t[4:], inputs['choke']['future'], '-', label=inputs['choke']['label'], color=inputs['choke']['color'])
    axs[1].legend(loc='upper right', prop={'size': 15})

    # gas lift
    axs1_twin = axs[1].twinx()
    axs1_twin.set_ylabel(inputs['gas lift']['ylabel'], fontsize=15, color=inputs['gas lift']['color'])
    axs1_twin.plot(t[4:], inputs['gas lift']['future'], '-', label=inputs['gas lift']['label'], color=inputs['gas lift']['color'])
    axs1_twin.legend(loc='upper right', prop={'size': 15})

    plt.show(block=False)
    if pause: 
        assert isinstance(pause_time, int), 'pause_time must be given as integer!'
        plt.pause(pause_time)
    plt.close()

    # -- SAVING FIGS -- # 
    if save_path is not None:   
        suffixes = ['.png', '.eps']

        if      exists(save_path.parent / (save_path.stem + '.png')) \
            or  exists(save_path.parent / (save_path.stem + '.eps')):
            name = input("Figure(s) with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if name != '': # _not_ aborting
                if name != 'y': # do _not_ want to override
                    save_path = save_path.parent / name

            else:
                save_path = None
            
        if save_path is not None:
            for suffix in suffixes:
                save_path = save_path.parent / (save_path.stem + suffix)
                fig.savefig(save_path, bbox_inches='tight')
        else:
            print("Figures were not saved.")


if __name__ == "__main__":
    t = np.linspace(0,10,num=10)

    inputs = {}
    inputs['title'] = 'Past, current and future inputs'
    inputs['choke'] = {'past': [0,1,2],
                       'current': [3],
                       'future': [4,5,6,7,8,9],
                       'ylabel': 'choke [%]',
                       'label': 'choke',
                       'color': 'tab:blue'}
    inputs['gas lift'] = {'past': [9,8,7],
                          'current': [6],
                          'future': [5,4,3,2,1,0],
                          'ylabel': 'gas lift [m^3/h]',
                          'label': 'gas lift',
                          'color': 'tab:purple'}
    
    outputs = {}
    outputs['title'] = 'Measured and predicted gas and oil rates'
    outputs['gas rate'] = {'past': [0,1,2],
                           'current': [3],
                           'future': [4,5,6,7,8,9],
                           'ylabel': 'gas rate [m^3/h]',
                           'label': 'gas rate',
                           'color': 'tab:orange'} 
    outputs['oil rate'] = {'past': [9,8,7],
                           'current': [6],
                           'future': [5,4,3,2,1,0],
                           'ylabel': 'oil rate lift [m^3/h]',
                           'label': 'oil rate',
                           'color': 'tab:green'} 

    t = np.linspace(0,10,num=10)
    plot_MPC_step(t, inputs, outputs, k=1, pause_time=30)
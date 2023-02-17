from pathlib import Path
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_RNNMPC(mpc=None, save_path=None, suffixes=None):
    """
    Plots data collected from a simulation of control by means of an RNNMPC
    
    See LSRMPC/src/plotting.py for """
    if mpc is None:
        return NotImplementedError
    
    num = len(mpc.simulated_u['full']['choke'])
    t = np.linspace(0, num * mpc.delta_t, num=num)
    fig, axes = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f'RNNMPC simulated {mpc.final_t // mpc.delta_t}', fontsize=23)

    # Plotting ground truth and predicted gas rates
    axes[0,0].set_title('Measured gas rate v. reference gas rate', fontsize=20)
    axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
    axes[0,0].plot(t, mpc.simulated_y['full']['gas rate'], '-', label='true gas rate', color='tab:orange')
    axes[0,0].plot(t[-mpc.final_t // mpc.delta_t:], mpc.full_refs['gas rate'], label='reference gas rate', color='tab:red')
    axes[0,0].legend(loc='best', prop={'size': 15})

    # Plotting ground truth and predicted oil rates
    axes[0,1].set_title('Measured oil rate v. reference oil rate', fontsize=20)
    axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
    axes[0,1].plot(t, mpc.simulated_y['full']['oil rate'], label='true oil rate', color='tab:orange')
    axes[0,1].plot(t[-mpc.final_t // mpc.delta_t:], mpc.full_refs['oil rate'], '-', label='reference oil rate', color='tab:red')
    axes[0,1].legend(loc='best', prop={'size': 15})

    # Plotting history of choke input
    axes[1,0].set_title('Input: choke opening', fontsize=20)
    axes[1,0].set_xlabel('time [s]', fontsize=15)
    axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
    axes[1,0].plot(t, mpc.simulated_u['full']['choke'], label='choke', color='blue')
    axes[1,0].legend(loc='best', prop={'size': 15})

    # Plotting history of gas lift rate input
    axes[1,1].set_title('Input: gas lift rate', fontsize=20)
    axes[1,1].set_xlabel('time [s]', fontsize=15)
    axes[1,1].set_ylabel('percent opening [m^3/h]', fontsize=15)
    axes[1,1].plot(t, mpc.simulated_u['full']['gas lift'], label='gas lift rate', color='blue')
    axes[1,1].legend(loc='best', prop={'size': 15})

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    plt.pause(10)
    plt.close()

    # Saving figs
    if save_path is not None:
        if suffixes is None: suffixes = ['.png', '.eps']  
        if not exists(save_path): # Safe to save; won't override
            for suffix in suffixes:
                # save_path = save_path.with_suffix(suffix)
                save_path = save_path.parent / (save_path.stem + suffix)
                fig.savefig(save_path, bbox_inches='tight')
                # fig.savefig(save_path.parent / (save_path.name + extra_suffix), bbox_inches='tight')

        else:
            name = input("Figure(s) with same filename already exists. Provide new name or \'y\' to overwrite ([enter] aborts save, file-endings are automatic): ")
            if name != '': # _not_ aborting
                if name == 'y': # want to override
                    for suffix in suffixes:
                        save_path = save_path.parent / (save_path.stem + suffix)
                        fig.savefig(save_path, bbox_inches='tight')
                        print("Figure(s) was(/were) overridden.")

                else: # assigning new name
                    for suffix in suffixes:
                        save_path = save_path.parent / (name + suffix)
                        fig.savefig(save_path, bbox_inches='tight')                
            else:
                print("Figure(s) was(/were) not saved.")

if __name__ == "__main__":
    plot_RNNMPC()
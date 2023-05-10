import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_LSRMPC(mpc=None, warm_start_cutoff: bool=True, plot_bias: bool=True, pause: bool=True, filter_output: bool=True):
    # -- FILTER -- #
    # Filter requirements.
    fs = 30.0       # sample rate, Hz
    cutoff_freq = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    order = 2       # sin wave can be approx represented as quadratic

    def butter_lowpass_filter(data, cutoff_freq, fs, order):
        nyq = 0.5 * fs # Nyquist frequency
        normal_cutoff = cutoff_freq / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    if not mpc:
        return NotImplementedError, 'Support for not providing mpc-object lacking. Implement path-capabilities.'
    else:
        t = np.array(mpc.t) # This array is just a list of floats, not a list of arrays, as the rest
        
        gas_rate = np.array(mpc.gas_rate_per_hr_vec)[:,0]
        gas_rate_ref = np.array(mpc.gas_rate_ref_vec)[:,0]
        gas_rate_bias = np.array(mpc.bias_gas)[:,0]
        
        oil_rate = np.array(mpc.oil_rate_per_hr_vec)[:,0]
        oil_rate_ref = np.array(mpc.oil_rate_ref_vec)[:,0]
        oil_rate_bias = np.array(mpc.bias_oil)[:,0]

        choke = np.array(mpc.choke_input)[:,0]
        gas_lift = np.array(mpc.gas_lift_input)[:,0]
        # TODO: Check if `choke_actual` is identical to `choke`
        choke_actual = np.array(mpc.choke_actual)[:,0] # TODO: Is this variable superfluous?


    """
    The time shift is performed because the initialization of the fmu is done with values
    for actuation ([50,0]) that are, although reasonable in and of themselves, not related to
    neither the control values nor the given references.

    To avoid plotting the resulting spikes in the beginning, a subsection is cut off.
    """

    # Filter the data, and plot both the original and filtered signals.
    if filter_output:
        gas_rate = butter_lowpass_filter(gas_rate, cutoff_freq, fs, order)
        oil_rate = butter_lowpass_filter(oil_rate, cutoff_freq, fs, order)

    # Plotting ground truth and predicted gas rates
    fig, axes = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f'LSRMPC simulated {mpc.final_time // mpc.delta_t} steps of {mpc.delta_t} [s] each. Total time is {mpc.final_time}', fontsize=23)

    axes[0,0].set_title('Measured gas rate v. reference gas rate', fontsize=20)
    axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
    axes[0,0].plot(t, gas_rate, '-', label='true gas rate', color='tab:orange')
    axes[0,0].plot(t[-len(gas_rate_ref):], gas_rate_ref, '--', label='reference gas rate', color='tab:red')
    if not warm_start_cutoff: axes[0,0].axvline(mpc.warm_start_t, color='tab:green')
    axes[0,0].legend(loc='best', prop={'size': 15})

    if plot_bias: 
        twin00 = axes[0,0].twinx()
        twin00.plot(t[-len(gas_rate_ref):], gas_rate_bias, label='bias gas rate', color='tab:olive')
        twin00.set_ylabel('gas rate bias (yk - yk_hat) [m^3/h]', color='tab:olive')
        twin00.tick_params(axis='y', color='tab:olive', labelcolor='tab:olive')
        twin00.spines['right'].set_color('tab:olive')
        max_lim = max(gas_rate_bias) + (2.5 * abs(max(gas_rate_bias)))
        min_lim = min(gas_rate_bias) - (0.5 * abs(min(gas_rate_bias)))
        twin00.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

    # Plotting ground truth and predicted oil rates
    axes[0,1].set_title('Measured oil rate v. reference oil rate', fontsize=20)
    axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
    axes[0,1].plot(t, oil_rate, label='true oil rate', color='tab:orange')
    axes[0,1].plot(t[-len(oil_rate_ref):], oil_rate_ref, '--', label='reference oil rate', color='tab:red')
    if not warm_start_cutoff: axes[0,1].axvline(mpc.warm_start_t, color='tab:green')
    axes[0,1].legend(loc='best', prop={'size': 15})

    if plot_bias: 
        twin01 = axes[0,1].twinx()
        twin01.plot(t[-len(gas_rate_ref):], oil_rate_bias, label='bias oil rate', color='tab:olive')
        twin01.set_ylabel('oil rate bias (yk - yk_hat) [m^3/h]', color='tab:olive')
        twin01.tick_params(axis='y', color='tab:olive', labelcolor='tab:olive')
        twin01.spines['right'].set_color('tab:olive')
        max_lim = max(oil_rate_bias) + (2.5 * abs(max(oil_rate_bias)))
        min_lim = min(oil_rate_bias) - (0.5 * abs(min(oil_rate_bias)))
        twin01.set_ylim(min_lim, max_lim) # Makes the plot less intrusive

    # Plotting history of choke input
    axes[1,0].set_title('Input: choke opening', fontsize=20)
    axes[1,0].set_xlabel('time [s]', fontsize=15)
    axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
    axes[1,0].plot(t, choke, label='choke', color='blue')
    if not warm_start_cutoff: axes[1,0].axvline(mpc.warm_start_t, color='tab:green')
    axes[1,0].legend(loc='best', prop={'size': 15})

    # Plotting history of gas lift rate input
    axes[1,1].set_title('Input: gas lift rate', fontsize=20)
    axes[1,1].set_xlabel('time [s]', fontsize=15)
    axes[1,1].set_ylabel('flow rate [m^3/h]', fontsize=15)
    axes[1,1].plot(t, gas_lift, label='gas lift rate', color='blue')
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

def plot_objective(obj_vals, sim_name: str, delta_t: int=10, pause: bool=True):# Plotting ground truth and predicted gas rates
    t = np.linspace(0, len(obj_vals) * delta_t, num=len(obj_vals))

    fig, ax = plt.plot()
    
    ax.set_title(f'Objective function of simulation: {sim_name}')
    ax.set_ylabel('objective function value', fontsize=15)
    ax.plot(t, obj_vals, '-', label='obj.', color='tab:blue')
    ax.legend(loc='best',prop={'size': 15})

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    if pause:
        plt.pause(10)
        plt.close()

    # Saving outsourced to main by returning fig    
    return fig

def plot_IO(gasrate, oilrate, choke, gaslift, time, delta_t, choke_act=None, gaslift_act=None, pause: bool=True):
    # Plotting ground truth and predicted gas rates
    fig, axes = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f'Simulated FMU for {time[-1] // delta_t} steps of {delta_t} [s] each. Total time is {time[-1]} [s]', fontsize=23)

    axes[0,0].set_title('Output: Gas rate', fontsize=20)
    axes[0,0].set_ylabel('gas rate [m^3/h]', fontsize=15)
    axes[0,0].plot(time, gasrate, '-', label='gas rate', color='tab:orange')
    axes[0,0].legend(loc='best', prop={'size': 15})

    # Plotting ground truth and predicted oil rates
    axes[0,1].set_title('Output: Oil rate', fontsize=20)
    axes[0,1].set_ylabel('oil rate [m^3/h]', fontsize=15)
    axes[0,1].plot(time, oilrate, label='oil rate', color='tab:orange')
    axes[0,0].legend(loc='best', prop={'size': 15})

    # Plotting history of choke input
    axes[1,0].set_title('Input: Choke opening', fontsize=20)
    axes[1,0].set_xlabel('time [s]', fontsize=15)
    axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
    axes[1,0].plot(time, choke, label='choke (act)', color='blue')

    # Plotting history of gas lift rate input
    axes[1,1].set_title('Input: gas lift rate', fontsize=20)
    axes[1,1].set_xlabel('time [s]', fontsize=15)
    axes[1,1].set_ylabel('flow rate [m^3/h]', fontsize=15)
    axes[1,1].plot(time, gaslift, label='gas lift rate (act)', color='blue')

    if choke_act is not None:
        # Plotting history of actually actuated choke input
        axes[1,0].set_title('Input: Choke opening', fontsize=20)
        axes[1,0].set_xlabel('time [s]', fontsize=15)
        axes[1,0].set_ylabel('percent opening [%]', fontsize=15)
        axes[1,0].plot(time, choke_act, '--', label='choke (meas)', color='red')
    
    if gaslift_act is not None:
        # Plotting history of gas lift rate input
        axes[1,1].set_title('Input: gas lift rate', fontsize=20)
        axes[1,1].set_xlabel('time [s]', fontsize=15)
        axes[1,1].set_ylabel('flow rate [m^3/h]', fontsize=15)
        axes[1,1].plot(time, gaslift_act, '--', label='gas lift rate (meas)', color='red')
        
    axes[1,0].legend(loc='best', prop={'size': 15})
    axes[1,1].legend(loc='best', prop={'size': 15})


    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    if pause:
        plt.pause(10)
        plt.close()

    # Saving outsourced to main by returning fig    
    return fig
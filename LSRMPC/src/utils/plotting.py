import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_LSRMPC(mpc=None):
    if not mpc:
        t = np.load('t.npy')
        oil_rate_per_hr_vec = np.load('oil_rate_per_hr_vec.npy')
        oil_rate_ref_vec = np.load('oil_rate_ref_vec.npy')

        gas_rate_per_hr_vec = np.load('gas_rate_per_hr_vec.npy')
        gas_rate_ref_vec = np.load('gas_rate_ref_vec.npy')

        choke_input = np.load('choke_input.npy')
        gas_lift_input = np.load('gas_lift_input.npy')
        choke_actual = np.load('choke_actual.npy')

        bias_gas = np.load('bias_gas.npy')
        bias_oil = np.load('bias_oil.npy')
    else:
        t = np.array(mpc.t) # This array is just a list of floats, not a list of arrays, as the rest
        oil_rate_per_hr_vec = np.array(mpc.oil_rate_per_hr_vec)[:,0]
        oil_rate_ref_vec = np.array(mpc.oil_rate_ref_vec)[:,0]

        gas_rate_per_hr_vec = np.array(mpc.gas_rate_per_hr_vec)[:,0]
        gas_rate_ref_vec = np.array(mpc.gas_rate_ref_vec)[:,0]

        choke_input = np.array(mpc.choke_input)[:,0]
        gas_lift_input = np.array(mpc.gas_lift_input)[:,0]
        choke_actual = np.array(mpc.choke_actual)[:,0]

        bias_gas = np.array(mpc.bias_gas)[:,0]
        bias_oil = np.array(mpc.bias_oil)[:,0]

    """
    The time shift is performed because the initialization of the fmu is done with values
    for actuation ([50,0]) that are, although reasonable in and of themselves, not related to
    neither the control values nor the given references.

    To avoid plotting the resulting spikes in the beginning, a subsection is cut off.
    """
    time_shift = 200 # Shift 200 steps
    t = t[time_shift:] - time_shift * 10 # Magic number because importing mpc.delta_t fails when mpc == None
    gas_rate_per_hr_vec = gas_rate_per_hr_vec[time_shift:]
    gas_rate_ref_vec = gas_rate_ref_vec[time_shift:]
    oil_rate_per_hr_vec = oil_rate_per_hr_vec[time_shift:]
    oil_rate_ref_vec = oil_rate_ref_vec[time_shift:]
    choke_input = choke_input[time_shift:]
    gas_lift_input = gas_lift_input[time_shift:]
    choke_actual = choke_actual[time_shift:]
    bias_gas = bias_gas[time_shift:]
    bias_oil = bias_oil[time_shift:]


    #Filter
    # Filter requirements.
    fs = 30.0       # sample rate, Hz
    cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    order = 2       # sin wave can be approx represented as quadratic

    def butter_lowpass_filter(data, cutoff, fs, order):
        nyq = 0.5 * fs # Nyquist frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    # Filter the data, and plot both the original and filtered signals.
    oil_rate_per_hr_vec = butter_lowpass_filter(oil_rate_per_hr_vec, cutoff, fs, order)
    gas_rate_per_hr_vec = butter_lowpass_filter(gas_rate_per_hr_vec, cutoff, fs, order)


    figure = plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(t, gas_rate_per_hr_vec, label = 'Gas rate', linewidth = 4)
    plt.plot(t, gas_rate_ref_vec, label = 'Reference signal', linewidth = 4, linestyle = "dashed")
    plt.title('Gas rate measurements', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Gas rate [m3/hr]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(t, oil_rate_per_hr_vec, label = 'Oil rate', linewidth = 4)
    plt.plot(t, oil_rate_ref_vec, label = 'Reference signal', linewidth = 4, linestyle = "dashed")
    plt.title('Oil rate measurements', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Oil rate [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    figure = plt.figure(2)
    plt.clf()
    plt.subplot(211)
    plt.plot(t, choke_input, label = 'Choke input', linewidth = 4)
    plt.plot(t, choke_actual, label = 'Actual choke opening', linewidth = 1)
    plt.title('Choke input', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Choke input [%]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    plt.subplot(212)
    plt.plot(t, gas_lift_input, label = 'Gas-lift rate', linewidth = 4)
    plt.title('Gas-lift rate input', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Gas-lift rate [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    figure = plt.figure(3)
    plt.subplot(211)
    plt.plot(t, bias_gas, label = 'Gas rate bias', linewidth = 4)
    plt.title('Gas rate bias', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Bias [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    plt.subplot(212)
    plt.plot(t, bias_oil, label = 'Oil rate bias', linewidth = 4)
    plt.title('Oil rate bias', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Bias [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')

    plt.show()

if __name__ == "__main__":
    plot_LSRMPC()
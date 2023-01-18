import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

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


t = t[200:]-2000
gas_rate_per_hr_vec = gas_rate_per_hr_vec[200:]
gas_rate_ref_vec = gas_rate_ref_vec[200:]
oil_rate_per_hr_vec = oil_rate_per_hr_vec[200:]
oil_rate_ref_vec = oil_rate_ref_vec[200:]
choke_input = choke_input[200:]
gas_lift_input = gas_lift_input[200:]
choke_actual = choke_actual[200:]
bias_gas = bias_gas[200:]
bias_oil = bias_oil[200:]


#Filter
# Filter requirements.
T = 5.0         # Sample Period
fs = 30.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

# sin wave
sig = np.sin(1.2*2*np.pi*t)
# Lets add some noise
noise = 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
data = sig + noise

def butter_lowpass_filter(data, cutoff, fs, order):
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
#plt.ylim((10000,14000))



plt.subplot(2,1,2)
plt.plot(t, oil_rate_per_hr_vec, label = 'Oil rate', linewidth = 4)
plt.plot(t, oil_rate_ref_vec, label = 'Reference signal', linewidth = 4, linestyle = "dashed")
plt.title('Oil rate measurements', fontsize=16)
plt.grid(linestyle='dotted', linewidth=1)
plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
plt.ylabel('Oil rate [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
plt.legend(fontsize = 'x-large', loc='lower right')
#plt.ylim((250,320))


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
#plt.plot(t, gas_lift_actual, label = 'Actual gas-lift rate', linewidth = 0.5)
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
plt.ylim((2000,8000))

plt.subplot(212)
plt.plot(t, bias_oil, label = 'Oil rate bias', linewidth = 4)
plt.title('Oil rate bias', fontsize=16)
plt.grid(linestyle='dotted', linewidth=1)
plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
plt.ylabel('Bias [m^3/hr]', fontdict=None, labelpad=None, fontsize=14)
plt.legend(fontsize = 'x-large', loc='lower right')
plt.ylim((200,260))


plt.show()
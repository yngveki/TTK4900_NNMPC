import numpy as np
import matplotlib.pyplot as plt

def unit_step(t, step_time):
    return np.heaviside(t - step_time, 1)

t = np.linspace(0, 10, num=100) # Generate time samples
step_time = 5 # Step time
y = unit_step(t, step_time) # Calculate the unit step function

plt.plot(t, y) # Plot the time series
plt.title("Unit Step Function")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.ylim([-0.2, 1.2]) # Set the y-axis limits
plt.show() # Show the plot
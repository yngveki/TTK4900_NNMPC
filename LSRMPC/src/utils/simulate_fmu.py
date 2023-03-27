# For further reference:
# https://jmodelica.org/pyfmi/tutorial.html#

# To generate step-response model

# from sre_constants import ASSERT_NOT
from pyfmi import load_fmu
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryFile

def _check_status(status_do_step):
    # Asserts whether a simulation step has been successful
    #
    # Input:
    # - status_do_step:     Status of a 1-step simulation
    print("Skjedd noe galt")
    #assert status_do_step < 1, f"FMI not OK, status code: {status_do_step} \n FMI_OK = 0 \n FMI_WARNING = 1 \n FMI_DISCARD = 2 \n FMI_ERROR = 3 \n FMI_FATAL = 4 \n FMI_PENDING = 5"

def init_model(model_path, start_time, final_time, delta_t, warm_start_t=1000, vals=[50,0]):
    # Initiates a FMU model at the given model_path
    #
    # Input:
    # - model_path:     path to relevant FMU
    # - start_time:     simulation start time
    # - final_time:     simulation final time
    #
    # Returns: Properly initiated model

    time = start_time
    model = load_fmu(model_path, log_level=7) #loglevel 7: log everything

    #Set SepticControl --> True
    model.set_boolean([536871484], [True])

    model.initialize(start_time, final_time, True) #Initialize the slave

    u1 = []
    u2 = []
    y1 = []
    y2 = []
    for i in range (warm_start_t // delta_t):
        model.set_real([3,4], vals)
        model.do_step(time, delta_t)
        time += delta_t
        u1.append(model.get('u[1]'))
        u2.append(model.get('u[2]'))
        y1.append(float(model.get('y[1]')))
        y2.append(float(model.get('y[2]')))

    # TODO: Verify units
    u1 = np.array(u1)
    u2 = np.array(u2)
    u_sim = np.hstack([u1,u2]) #u_sim er en vektor 80x2 

    y1 = np.array(y1)
    y2 = np.array(y2)
    y1 = np.reshape(y1, (len(y1), 1))
    y2 = np.reshape(y2, (len(y2), 1)) 
    y_sim = np.hstack([y1,y2]) #y_sim er en vektor 80x2 

    return model, u_sim, y_sim

def simulate_singlewell_step(model, time, delta_t, Uk, time_series=None):
    # Simulates one step of the SingleWell model, from time to time + delta_t.
    #
    # Inputs:
    # - model:          FMU in the form of a model object
    # - time:           current timestep from which we simulate 1 step ahead
    # - delta_t:        time interval per timestep
    # - Uk:             vector of actuations for current time step, shape: n_MV x 1
    # - time_series:    a time series that tracks for which timesteps we have simulated. For plotting purposes. (OPTIONAL)
    #
    # Returns: the two CVs of interest from the model. Optionally also the updated time series.

    # Apply input
    model.set_real([3,4], [Uk[0], Uk[1]]) #check inputs ##############BYTTE INPUT INDEXER PGA NY FMU

    # Perform 1-step simulation
    _ = model.do_step(time, delta_t) #Do step delta_t from current time, maybe add status_step

    # TODO: Verify units
    # Get output
    gas_rate = float(model.get('y[1]')) # _Should_ be m3/hr
    oil_rate = float(model.get('y[2]')) # _Should_ be m3/hr
    u1 = float(model.get('u[1]'))
    u2 = float(model.get('u[2]'))
    choke_opening = model.get('choke.opening')
    gas_lift_current= float(model.get('y[4]'))

    return [gas_rate, oil_rate, u1, u2, choke_opening, gas_lift_current]
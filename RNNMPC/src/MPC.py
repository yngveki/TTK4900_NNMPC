#!/usr/bin/env python3

import casadi as cs

from utils.simulate_fmu import init_model, simulate_singlewell_step
from neuralnetwork import NeuralNetwork

class RNNMPC:

    def __init__(self, nn_path):
        """
        Takes in given paths to setup the framework around the OCP
        """
        
        # Initialized variables to be filled out during runtime
        self.fmu = None

        self.simulated_u = {}
        self.simulated_u['init'] = [] # Will keep data from warm_start
        self.simulated_u['sim'] = []  # Will keep data from control loop
        self.simulated_u['full'] = [] # Concatenates the two above
        self.simulated_y = {}
        self.simulated_y['init'] = [] # Will keep data from warm_start
        self.simulated_y['sim'] = []  # Will keep data from control loop
        self.simulated_y['full'] = [] # Concatenates the two above

        # Initialize basics of OCP
        self.args = {}  # TODO
        self.opts = {}  # TODO
        nlp = {'x': ..., 'p': ..., 'f': ..., 'g': ...}  # TODO
        self.solver = cs.nlpsol('solver', 'ipopt', nlp, self.opts)

        # Load neural network model to assign weights and biases as coefficients
        # to equality constraint in OCP 
        
        # Load model
        self.model = NeuralNetwork()
        self.model.load(nn_path)

        # Extract weights and biases
        weights, biases = self.model.extract_coefficients()


        # Set weights and biases as coefficients of OCP
        # TODO

    def warm_start(self, fmu_path, warm_start_t=1000):
        """
        Simulates the fmu for a few steps to ensure defined state before optimization loop
        """
        self.fmu, \
        self.simulated_u['init'], \
        self.simulated_y['init'] = init_model(fmu_path, 
                                                        self.start_time, 
                                                        self.final_time, # Needed for initialization, but different from warm start time
                                                        self.delta_t, 
                                                        self.Hp,
                                                        warm_start_t)

    def update_OCP(self):
        return NotImplementedError

    def solve_OCP(self):
        res = self.solver(**self.args)
        self.u_opt = res[0:1] # TODO: What will be correct format?

    def iterate_system(self):
        gas_rate_k, oil_rate_k, \
        choke_act_k, gas_lift_act_k, \
        _, _ = simulate_singlewell_step(self.model, 
                                        self.time, 
                                        self.delta_t, 
                                        self.U_next) # measurement from FMU, i.e. result from previous actuation

        self.simulated_y['sim'].append([gas_rate_k, oil_rate_k])
        self.simulated_u['sim'].append([choke_act_k, gas_lift_act_k])                                                                           
        
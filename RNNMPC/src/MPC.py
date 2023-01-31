#!/usr/bin/env python3

import casadi as cs
from yaml import safe_load

from utils.simulate_fmu import init_model, simulate_singlewell_step
from neuralnetwork import NeuralNetwork

class RNNMPC:

    def __init__(self, 
                nn_path,
                config_path):
        """
        Takes in given paths to setup the framework around the OCP
        """
        
        # -- Initialization of variables for later availability -- #
        self.fmu = None

        self.simulated_u = {}
        self.simulated_u['init'] = [] # Will keep data from warm_start
        self.simulated_u['sim'] = []  # Will keep data from control loop
        self.simulated_u['full'] = [] # Concatenates the two above
        self.simulated_y = {}
        self.simulated_y['init'] = [] # Will keep data from warm_start
        self.simulated_y['sim'] = []  # Will keep data from control loop
        self.simulated_y['full'] = [] # Concatenates the two above

        # -- config parameters -- #
        configs = self._read_yaml(config_path)

        # Horizons
        self.Hu = configs['TUNING_PARAMETERS']['Hu']
        self.Hp = configs['TUNING_PARAMETERS']['Hp']

        # Weights
        self.Q = configs['TUNING_PARAMETERS']['Q']
        self.P = configs['TUNING_PARAMETERS']['P']
        self.rho = configs['TUNING_PARAMETERS']['rho']

        # Constraints
        self.ylb = configs['TUNING_PARAMETERS']['ylb']
        self.yub = configs['TUNING_PARAMETERS']['yub']
        self.ulb = configs['TUNING_PARAMETERS']['ulb']
        self.uub = configs['TUNING_PARAMETERS']['uub']
        self.dulb = configs['TUNING_PARAMETERS']['dulb']
        self.duub = configs['TUNING_PARAMETERS']['duub']
        self.elb = configs['TUNING_PARAMETERS']['elb']
        self.eub = configs['TUNING_PARAMETERS']['eub']

        # Timekeeping
        self.delta_t = configs['RUNNING_PARAMETERS']['delta_t']
        self.final_t = configs['RUNNING_PARAMETERS']['final_t']
        self.t = 0

        # -- Set up framework for OCP -- #
        self.args = {}  # TODO
        self.opts = {}  # TODO
        nlp = {'x': ..., 'p': ..., 'f': ..., 'g': ...}  # TODO
        self.solver = cs.nlpsol('solver', 'ipopt', nlp, self.opts)

        # -- Load neural network model -- #        
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
                                            start_time = self.t, 
                                            final_time = self.final_t, # Needed for initialization, but different from warm start time
                                            delta_t = self.delta_t,
                                            warm_start_t=2000) # TODO: Figure out adequate value

    def update_OCP(self):
        return NotImplementedError

    def solve_OCP(self):
        res = self.solver(**self.args)
        self.u_opt = res[0:1] # TODO: What will be correct format?

    def iterate_system(self):
        gas_rate_k, oil_rate_k, \
        choke_act_k, gas_lift_act_k, \
        _, _ = simulate_singlewell_step(self.model, 
                                        self.t, 
                                        self.delta_t, 
                                        self.u_opt) # measurement from FMU, i.e. result from previous actuation

        self.simulated_y['sim'].append([gas_rate_k, oil_rate_k])
        self.simulated_u['sim'].append([choke_act_k, gas_lift_act_k])   

    def merge_sim_data(self):
        """
        Convenience function to make the full timeseries more convenient
        """
        # TODO: If this is done at init, will the ['full'] array update dynamically? -> TEST
        self.simulated_u['full'] = self.simulated_u['init'] + self.simulated_u['sim']
        self.simulated_y['full'] = self.simulated_y['init'] + self.simulated_y['sim']                                                                        
    
    
    # --- Private funcs --- #
    def _read_yaml(self, file_path):
        """
        Returns content of a YAML-file at given path
        """
        
        with open(file_path, "r") as f:
            return safe_load(f)
        
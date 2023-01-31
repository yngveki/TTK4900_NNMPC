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

        # Set parameters of MPC

        configs = self._read_yaml(config_path)

        # -- config used during updates inside control loop -- #
        self.n_MV = configs['SYSTEM_PARAMETERS']['n_MV']         # number of input variables - given from system (p.426)
        self.n_CV = configs['SYSTEM_PARAMETERS']['n_CV']         # number of outputs - given from system (p. 423) 

        self.Hu = configs['TUNING_PARAMETERS']['Hu']             # control horizon - design parameter (p. 416)
        self.Hp = configs['TUNING_PARAMETERS']['Hp']             # prediction horizon - design parameter (p. 417)
        self.Hw = configs['TUNING_PARAMETERS']['Hw']             # time step from which prediction horizon effectively starts
        
        self.rho_h = np.array(configs['TUNING_PARAMETERS']['rho_h'], dtype=int, ndmin=2).T
        self.rho_l = np.array(configs['TUNING_PARAMETERS']['rho_l'], dtype=int, ndmin=2).T

        self.du_over_bar = configs['TUNING_PARAMETERS']['du_over_bar']      # Upper limit of deltaMVs
        self.du_under_bar = configs['TUNING_PARAMETERS']['du_under_bar']    # Lower limit of deltaMVs
        self.e_over_bar = configs['TUNING_PARAMETERS']['e_over_bar']        # Upper limit of slack variables epsilon
        self.e_under_bar = configs['TUNING_PARAMETERS']['e_under_bar']      # Lower limit of slack variables epsilon
        
        # -- config used only during initialization -- #
        n_eh = self.n_CV                                         # Same as "ny_over_bar"; amount of CVs that have upper limits
        n_el = self.n_CV                                         # Same as "ny_under_bar"; amount of CVs that have lower limits
        
        P_bar = configs['TUNING_PARAMETERS']['P_bar']       # Weight for actuations 
        Q_bar = configs['TUNING_PARAMETERS']['Q_bar']       # Weight for errors from ref
        
        y_over_bar = configs['TUNING_PARAMETERS']['y_over_bar']             # Upper limit of CVs
        y_under_bar = configs['TUNING_PARAMETERS']['y_under_bar']           # Lower limit of CVs
        u_over_bar = configs['TUNING_PARAMETERS']['u_over_bar']             # Upper limit of MVs
        u_under_bar = configs['TUNING_PARAMETERS']['u_under_bar']           # Lower limit of MVs

        # -- time-keeping -- #
        self.delta_t = configs['RUNNING_PARAMETERS']['delta_t']
        self.final_time = configs['RUNNING_PARAMETERS']['final_time']
        self.time = 0
        self.start_time = self.time
        num_steps = self.final_time // self.delta_t

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
    
    
    # --- Private funcs --- #
    def _read_yaml(self, file_path):
        """
        Returns content of a YAML-file at given path
        """
        
        with open(file_path, "r") as f:
            return safe_load(f)
        
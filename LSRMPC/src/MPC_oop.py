import numpy as np
from qpsolvers import solve_qp
import matrix_generation
from yaml import safe_load
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import simulate_fmu

class MPC:

    def __init__(self, config_path, S_paths):
        """
        Takes in given paths to setup configuration parameters and constant matrices
        """
        self.__N = 100 # TODO: Should be defined more robust
        self.__OC_solved = False # TODO: Possibly redundant

        # -- Setting config -- #
        configs = self._read_yaml(config_path)

        # -- -- config used only during initialization -- -- #
        n_MV = configs['SYSTEM_PARAMETERS']['n_MV']         # number of input variables - given from system (p.426)
        self.n_CV = configs['SYSTEM_PARAMETERS']['n_CV']         # number of outputs - given from system (p. 423) 

        P_bar = configs['TUNING_PARAMETERS']['P_bar']       # Weight for actuations 
        Q_bar = configs['TUNING_PARAMETERS']['Q_bar']       # Weight for errors from ref
        n_eh = self.n_CV                                         # Same as "ny_over_bar"; amount of CVs that have upper limits
        n_el = self.n_CV                                         # Same as "ny_under_bar"; amount of CVs that have lower limits
        self.rho_h_parameter = configs['TUNING_PARAMETERS']['rho_h'], (1,1)
        self.rho_l_parameter = configs['TUNING_PARAMETERS']['rho_l'], (1,1)
        self.rho_h = np.vstack((self.rho_h_parameter[0][0], self.rho_h_parameter[0][1])) # Weight for slack variable eps_h
        self.rho_l = np.vstack((self.rho_l_parameter[0][0], self.rho_l_parameter[0][1])) # Weight for slack variable eps_l

        self.z_bar_lb = configs['TUNING_PARAMETERS']['z_bar_lb']
        self.z_bar_ub = configs['TUNING_PARAMETERS']['z_bar_ub'] # shape

        self.y_over_bar = configs['TUNING_PARAMETERS']['y_over_bar']        # Upper limit of CVs
        self.y_under_bar = configs['TUNING_PARAMETERS']['y_under_bar']      # Lower limit of CVs
        self.u_over_bar = configs['TUNING_PARAMETERS']['u_over_bar']        # Upper limit of MVs
        self.u_under_bar = configs['TUNING_PARAMETERS']['u_under_bar']      # Lower limit of MVs

        # -- -- config used also during updates -- -- #
        self.Hu = configs['TUNING_PARAMETERS']['Hu']             # control horizon - design parameter (p. 416)
        self.Hp = configs['TUNING_PARAMETERS']['Hp']             # prediction horizon - design parameter (p. 417)
        self.Hw = configs['TUNING_PARAMETERS']['Hw']             # time step from which prediction horizon effectively starts

        # -- Reading in the step response model -- #
        self.Sijs = np.array([self._S_xx_append(self.Hp, siso) for siso in self._read_S(S_paths)])
        
        # -- Build constant matrices -- #
        self.Theta = matrix_generation.get_Theta(self.Sijs, self.Hp, self.Hu, self.Hw)
        self.Psi = matrix_generation.get_Psi(self.Sijs, self.Hp, self.Hu, self.Hw, self.__N)
        self.Upsilon = matrix_generation.get_Upsilon(self.Sijs, self.Hp, self.Hw)
        
        # -- -- Build minimization problem matrix Hd -- -- #
        self.Q_y = matrix_generation.get_Q(self.Q_bar, self.Hp, self.Hw)    # shape: (n_CV * (Hp - Hw + 1)) x (n_CV * (Hp - Hw + 1)).
        self.P = matrix_generation.get_P(self.P_bar, self.Hu)               # shape: (n_MV * Hu) x (n_MV * Hu).

        self.Hd = np.zeros(((n_MV * Hu) + n_eh + n_el, (n_MV * Hu) + n_eh + n_el))
        self.Hd[:(n_MV * Hu), :(n_MV * Hu)] = Theta.T @ Q_y @ Theta + P # Hd_bar
        self.Hd *= 2

        # -- -- Build inequality system matrix Ad -- -- #
        self.G = matrix_generation.get_G(Hp, Hw, n_CV)
        self.F = matrix_generation.get_F(Hu, n_MV)
        self.Kinv = np.linalg.inv(matrix_generation.get_K(Hu, n_MV))
        self.Mh = matrix_generation.get_Mh(Hp, Hw, n_eh)
        self.Ml = matrix_generation.get_Ml(Hp, Hw, n_el)

        vec1 = np.hstack((G@Theta, -Mh, -Ml))
        FKinv = self.F @ self.Kinv
        zeros22 = np.zeros((FKinv.shape[0], Mh.shape[1]))
        zeros23 = np.zeros((FKinv.shape[0], Ml.shape[1]))
        vec2 = np.hstack((FKinv, zeros22, zeros23))
        self.Ad = np.vstack((vec1, vec2))
        
        # -- -- Build the rest -- -- #
        # Necessary for variable matrices later
        self.gamma = -2 * (Theta.T @ Q_y)
        self.g = matrix_generation.get_g(Hp, Hw, n_CV, y_over_bar, y_under_bar)
        self.f = matrix_generation.get_f(Hu, n_MV, u_over_bar, u_under_bar)
        self.Gamma = matrix_generation.get_Gamma(Hu, n_MV)
        self.FKinvGamma = FKinv @ Gamma

        # -- Set up inputs and outputs -- #
        # -- -- Set up inputs -- -- #
        U_tilde_prev = np.zeros((n_MV, 1))            # U_tilde(k - 1), p.36 in Kufoalor
        U_tilde_N = np.zeros((n_MV, 1))               # U_tilde(k - Nj), where Nj == N
        dU_tilde_prev = np.zeros((N-Hw-1, n_MV))
        u_prev = np.zeros((n_MV, 1))
        U = np.zeros((Hu*n_MV, 1))
        U_next = np.zeros((2,1))

        # -- -- Set up outputs -- -- #

    def warm_start(self, fmu_path):
        """
        Simulates the fmu for a few steps to ensure defined state
        """
        self.model, self.u_sim, self.y_sim = simulate_fmu.init_model(fmu_path, start_time, final_time, delta_t, Hp)
        

    def update_matrices(self):
        """
        Updates all matrices that are time-varying
        """
        self.V = matrix_generation.get_V_matrix(y_prev, y_hat_k_minus_1, self.Hp, self.Hw)

    def update_OCP(self):
        """
        Sets the OCP matrices and constraints. Intended as an update after matrices and/or
        constraints have been updated for increased time, such that the OCP is up to date
        """
        ...

    def solve_OCP(self):
        """
        Performs the solving of the OCP
        """
        ...
        # TODO: Solve OCP
        self.__OC_solved = True

    def iterate_system(self):
        """
        Applies optimal control and updates values input and output values' timestep
        accordingly
        """
        assert self.__OC_solved == True, "System values are outdated. Cannot iterate system."

        ...
        
        self.__OC_solved = False
        
    def step_input(self, steps):
        """
        Adds steps to the MPC's reference T at defined times.
        
        :param steps: list of tuples, where each tuple holds timestep and a tuple of increments
        """


    # --- Private funcs --- #
    def _read_yaml(self, file_path):
        # Reads the config-file of YAML-format at designated file_path and returns 
        # a dictionary containing the configs
        #
        # Input:
        # - file_path: file path relative to current script
        # 
        # Returns: Dictionary containing all config-values

        with open(file_path, "r") as f:
            return safe_load(f)

    def _S_xx_append(P, S_xx):
        S_xx_steady = S_xx[(len(S_xx)-1)]
        
        for i in range(P-len(S_xx)):
            S_xx = np.append(S_xx, S_xx_steady)

        return S_xx

    def _read_S(rel_S_paths):
        # Reads data file containing the S-matrix
        #
        # Input:
        # - file_path: file path relative to current script
        # 
        # Returns: S matrix

        Sijs = []
        for key in rel_S_paths:
            # Sijs.append(S_xx_append(N, np.load(Path(__file__).parent / rel_path)))
            Sijs.append(np.load(Path(__file__).parent / rel_S_paths[key]))

        return np.array(Sijs)
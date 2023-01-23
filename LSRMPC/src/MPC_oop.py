# Library imports
import numpy as np
from qpsolvers import solve_qp
import matrix_generation
from yaml import safe_load
from pathlib import Path
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

# Custom imports
from simulate_fmu import init_model, simulate_singlewell_step
from utils.references import References

from custom_timing import Timer
stopwatch = Timer()

class MPC:

    __N = 100 # The length of each SISO step response model once all are made equally long

    def __init__(self, 
                config_path,
                S_paths, 
                ref_path):
        """
        Takes in given paths to setup configuration parameters and constant matrices
        """

        self.model = None # Initialization

        #### ----- Setting config ----- ####
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
        
        # -- reference -- #
        self.refs = References(ref_path)
        self.y_ref = self.refs.curr_ref

        #### ----- Reading in the step response model ----- ####
        self.Sijs = np.array([self._S_xx_append(self.Hp, siso) for siso in self._read_S(S_paths)])
        
        #### ----- Build constant matrices ----- ####
        self.Theta = matrix_generation.get_Theta(self.Sijs, self.Hp, self.Hu, self.Hw)
        self.Psi = matrix_generation.get_Psi(self.Sijs, self.Hp, self.Hu, self.Hw, self.__N)
        self.Upsilon = matrix_generation.get_Upsilon(self.Sijs, self.Hp, self.Hw)

        # -- Build minimization problem matrix Hd -- #
        Q_y = matrix_generation.get_Q(Q_bar, self.Hp, self.Hw)    # shape: (n_CV * (Hp - Hw + 1)) x (n_CV * (Hp - Hw + 1)).
        P = matrix_generation.get_P(P_bar, self.Hu)               # shape: (n_MV * Hu) x (n_MV * Hu).

        self.Hd = np.zeros(((self.n_MV * self.Hu) + n_eh + n_el, (self.n_MV * self.Hu) + n_eh + n_el))
        self.Hd[:(self.n_MV * self.Hu), :(self.n_MV * self.Hu)] = self.Theta.T @ Q_y @ self.Theta + P # Hd_bar
        self.Hd *= 2

        # -- Build inequality system matrix Ad -- #
        self.G = matrix_generation.get_G(self.Hp, self.Hw, self.n_CV)
        F = matrix_generation.get_F(self.Hu, self.n_MV)
        self.Kinv = np.linalg.inv(matrix_generation.get_K(self.Hu, self.n_MV))
        Mh = matrix_generation.get_Mh(self.Hp, self.Hw, n_eh)
        Ml = matrix_generation.get_Ml(self.Hp, self.Hw, n_el)

        vec1 = np.hstack((self.G @ self.Theta, -Mh, -Ml))
        FKinv = F @ self.Kinv
        zeros22 = np.zeros((FKinv.shape[0], Mh.shape[1]))
        zeros23 = np.zeros((FKinv.shape[0], Ml.shape[1]))
        vec2 = np.hstack((FKinv, zeros22, zeros23))
        self.Ad = np.vstack((vec1, vec2))
        
        # -- Build the rest -- #
        # Necessary for variable matrices later
        self.gamma = -2 * (self.Theta.T @ Q_y)
        self.g = matrix_generation.get_g(self.Hp, self.Hw, self.n_CV, y_over_bar, y_under_bar)
        self.f = matrix_generation.get_f(self.Hu, self.n_MV, u_over_bar, u_under_bar)
        self.Gamma = matrix_generation.get_Gamma(self.Hu, self.n_MV)
        self.FKinvGamma = FKinv @ self.Gamma

        #### ----- Set up inputs and outputs ----- ####
        # -- Set up inputs -- #
        self.U_tilde_prev = np.zeros((self.n_MV, 1))            # U_tilde(k - 1), p.36 in Kufoalor
        self.dU_tilde_prev = np.zeros(((self.__N - self.Hw - 1) * self.n_MV, 1))
        self.u_prev_act = np.zeros((self.n_MV, 1))
        self.u_prev_meas = np.zeros((self.n_MV, 1))

        # -- Set up outputs -- #
        self.y_hat = np.zeros((self.n_CV, 1))
        self.y_hat_k_minus_1 = np.zeros((2,1))
        self.y_prev = np.zeros((self.n_CV, 1))

        #### ----- Initialize data structures to hold data that may be plotted ----- ####
        self.gas_rate_per_hr_vec = []
        self.oil_rate_per_hr_vec = []
        self.gas_rate_ref_vec = []
        self.oil_rate_ref_vec = []
        self.choke_input = []
        self.gas_lift_input = []
        self.choke_actual = []
        self.gas_lift_actual = []
        self.bias_gas = []
        self.bias_oil = []
        self.t = []

    def warm_start(self, fmu_path, warm_start_t=1000):
        """
        Simulates the fmu for a few steps to ensure defined state before optimization loop
        """
        self.model, self.u_sim, self.y_sim = init_model(fmu_path, 
                                                        self.start_time, 
                                                        self.final_time, # Needed for initialization, but different from warm start time
                                                        self.delta_t, 
                                                        self.Hp,
                                                        warm_start_t)
        self.y_prev[0] = self.y_sim[-1][0]
        self.y_prev[1] = self.y_sim[-1][1]
        self.y_hat[0] = 0
        self.y_hat[1] = 0

        self.y_hat_k_minus_1[0] = self.y_sim[-1][0]
        self.y_hat_k_minus_1[1] = self.y_sim[-1][1]

        self.U_tilde_prev[0] = self.u_sim[-1][0] 
        self.U_tilde_prev[1] = self.u_sim[-1][1] 

        # Will produce only zeros in dU_tilde_prev since self.u_sim is constant. Kept for generality
        u_sim_flip = np.flip(self.u_sim)
        for i in range(self.n_MV):
            for j in range(self.__N - self.Hw - 1):
                #Flipper u_sim slik at den forrige implementerte inputen er først i u_sim_flip: dU_tilde = [u(k-1), u(k-2), ....]
                self.dU_tilde_prev[i * (self.__N - self.Hw - 1) + j] = u_sim_flip[j+1][i] - u_sim_flip[j][i]

        y_hat_init = self.Psi @ self.dU_tilde_prev + self.Upsilon @ self.U_tilde_prev
        self.y_hat_k_minus_1[0] = y_hat_init[0]
        self.y_hat_k_minus_1[1] = y_hat_init[self.Hp - self.Hw + 1]

    def update_matrices(self):
        """
        Updates all matrices that are time-varying
        """
        self.V = matrix_generation.get_V_matrix(self.y_prev, self.y_hat_k_minus_1, self.Hp, self.Hw)
        Lambda_d = self.Psi @ self.dU_tilde_prev + self.Upsilon @ self.U_tilde_prev + self.V        
        self.T = matrix_generation.get_T(self.refs.curr_ref, self.n_CV, self.Hp, self.Hw) #TODO: Alter the way this works. Don't need to update at every timestep when y_ref is constant. With the "steps" function, no update should be required at all, really
        
        zeta = self.T - Lambda_d

        self.gd = np.vstack(((zeta.T @ self.gamma.T).T, 
                              self.rho_h, 
                              self.rho_l)).ravel()
        self.bd = np.vstack((-self.G @ Lambda_d + self.g, 
                             -self.FKinvGamma @ self.U_tilde_prev + self.f))

    def update_OCP(self):
        """
        Sets the OCP matrices and constraints. Intended as an update after matrices and/or
        constraints have been updated for increased time, such that the OCP is up to date
        """
        # TODO:
        # May be possible to do this dynamically, isntead of reinitializing at every step, by setAttr
        # https://support.gurobi.com/hc/en-us/community/posts/360054720932-Updating-the-RHS-and-LHS-of-specific-constraints-Python-
        # ^ if you want to look into it
        #
        # Alternatively use getConstrs and chgCoeff for all constraints

        # #### ----- Initialize model for solving with gurobi ----- ####
        self.m = gp.Model("qp")
        self.m.Params.LogToConsole = 0

        variables = []
        for i in range(self.Hu):
            variables.append(self.m.addVar(lb=self.du_under_bar[0], ub=self.du_over_bar[0], name="dU_1"))

        for i in range(self.Hu):
            variables.append(self.m.addVar(lb=self.du_under_bar[1], ub=self.du_over_bar[1], name="dU_2"))

        for i in range(4):
            variables.append(self.m.addVar(lb=self.e_under_bar, ub=self.e_over_bar, name="slack_var"))

        self.m.addMConstr(self.Ad, variables, "<=", self.bd)
        self.m.setMObjective(self.Hd, self.gd, 0, None, None, None, GRB.MINIMIZE)

        self.m.update()

    def solve_OCP(self):
        """
        Performs the solving of the OCP and stores the next optimal control actions.
        """
        self.m.optimize()

        dU = np.array([v.X for v in self.m.getVars()], dtype=float, ndmin=2).T
        dU = dU[:self.Hu * 2] # Indexing dU since we only want values for actuations, not slack variables
        
        U = self.Kinv @ (self.Gamma @ self.U_tilde_prev + dU) # TODO: Might need class-wide scope if needed for plotting
        self.U_next = [U[0], U[self.Hu]]

        self.y_hat = self.Psi @ self.dU_tilde_prev + self.Upsilon @ self.U_tilde_prev + self.Theta @ dU

    def iterate_system(self):
        """
        Applies optimal control and updates values input and output values' timestep
        accordingly
        """
        # stopwatch.start()
        self.y_prev[0], self.y_prev[1], \
        self.u_prev_act[0], self.u_prev_act[1], \
        self.u_prev_meas[0], self.u_prev_meas[1] = simulate_singlewell_step(self.model, 
                                                                            self.time, 
                                                                            self.delta_t, 
                                                                            self.U_next) # measurement from FMU, i.e. result from previous actuation
        # --- Roll the input, and set the newest input at the end --- #
        self.u_sim = np.roll(self.u_sim, [-1, -1]) # put newest input at end of array
        self.u_sim[-1, 0] = self.u_prev_act[0]
        self.u_sim[-1, 1] = self.u_prev_act[1]

        self.U_tilde_prev[0] = self.u_prev_act[0]
        self.U_tilde_prev[1] = self.u_prev_act[1]

        # --- Roll the change input, and set the newest change in input at the end --- #
        self.dU_tilde_prev = np.roll(self.dU_tilde_prev, 1)
        self.dU_tilde_prev[0] = self.u_sim[-1][0] - self.u_sim[-2][0]
        self.dU_tilde_prev[self.__N - self.Hw - 1] = self.u_sim[-1][1] - self.u_sim[-2][1]

        # --- Collect the first predicted output --- #
        self.y_hat_k_minus_1[0] = self.y_hat[0]
        self.y_hat_k_minus_1[1] = self.y_hat[self.Hp - self.Hw + 1]

        # --- Set the newest measurement at the end --- #
        self.y_sim = np.roll(self.y_sim, [-1, -1]) # put newest measurement at end of array
        self.y_sim[-1, 0] = self.y_prev[0]
        self.y_sim[-1, 1] = self.y_prev[1]
        # stopwatch.stop()
        
        # --- Update plotting-data --- #
        self.gas_rate_per_hr_vec.append(self.y_prev[0])
        self.oil_rate_per_hr_vec.append(self.y_prev[1])
        self.gas_rate_ref_vec.append(self.T[0])
        self.oil_rate_ref_vec.append(self.T[self.Hp])


        self.choke_input.append(self.u_prev_act[0])
        self.gas_lift_input.append(self.u_prev_act[1])
        self.choke_actual.append(self.u_prev_meas[0])
        self.gas_lift_actual.append(self.u_prev_meas[1]*1000/24)
        self.bias_gas.append(self.V[0])
        self.bias_oil.append(self.V[-1])

        self.t.append(self.time)

        self.time += self.delta_t
        self.refs.curr_time += self.delta_t


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

    def _S_xx_append(self, P, S_xx):
        S_xx_steady = S_xx[(len(S_xx)-1)]
        
        for i in range(P-len(S_xx)):
            S_xx = np.append(S_xx, S_xx_steady)

        return S_xx

    def _read_S(self, rel_S_paths):
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

if __name__ == "__main__":
    print("this is for testing")
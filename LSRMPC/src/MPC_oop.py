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
    def __init__(self, config_path):
        """
        Takes in given paths to setup configuration parameters and constant matrices
        """
        # -- Setting config -- #
        configs = self._read_yaml(config_path)

        self.n_MV = configs['SYSTEM_PARAMETERS']['n_MV']         # number of input variables - given from system (p.426)
        self.n_CV = configs['SYSTEM_PARAMETERS']['n_CV']         # number of outputs - given from system (p. 423) 
        self.Hu = configs['TUNING_PARAMETERS']['Hu']             # control horizon - design parameter (p. 416)
        self.Hp = configs['TUNING_PARAMETERS']['Hp']             # prediction horizon - design parameter (p. 417)
        self.Hw = configs['TUNING_PARAMETERS']['Hw']             # time step from which prediction horizon effectively starts

        self.P_bar = configs['TUNING_PARAMETERS']['P_bar']       # Weight for actuations 
        self.Q_bar = configs['TUNING_PARAMETERS']['Q_bar']       # Weight for errors from ref
        self.n_eh = n_CV                                         # Same as "ny_over_bar"; amount of CVs that have upper limits
        self.n_el = n_CV                                         # Same as "ny_under_bar"; amount of CVs that have lower limits
        self.rho_h_parameter = configs['TUNING_PARAMETERS']['rho_h'], (1,1)
        self.rho_l_parameter = configs['TUNING_PARAMETERS']['rho_l'], (1,1)
        self.rho_h = np.vstack((rho_h_parameter[0][0], rho_h_parameter[0][1])) # Weight for slack variable eps_h
        self.rho_l = np.vstack((rho_l_parameter[0][0], rho_l_parameter[0][1])) # Weight for slack variable eps_l

        self.z_bar_lb = configs['TUNING_PARAMETERS']['z_bar_lb']
        self.z_bar_ub = configs['TUNING_PARAMETERS']['z_bar_ub'] # shape

        self.y_over_bar = configs['TUNING_PARAMETERS']['y_over_bar']        # Upper limit of CVs
        self.y_under_bar = configs['TUNING_PARAMETERS']['y_under_bar']      # Lower limit of CVs
        self.u_over_bar = configs['TUNING_PARAMETERS']['u_over_bar']        # Upper limit of MVs
        self.u_under_bar = configs['TUNING_PARAMETERS']['u_under_bar']      # Lower limit of MVs

        # -- Setting constant matrices -- #
        
    def update_matrices(self):
        """
        Updates all matrices that are time-varying
        """
        ...

    def update_OCP(self):
        ...

    def solve_OCP(self):
        ...

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

# ----- FUNCTIONS ----- #

def read_yaml(file_path):
    # Reads the config-file of YAML-format at designated file_path and returns 
    # a dictionary containing the configs
    #
    # Input:
    # - file_path: file path relative to current script
    # 
    # Returns: Dictionary containing all config-values

    with open(file_path, "r") as f:
        return safe_load(f)

def build_full_S(Sijs, n_CV, n_MV, Hu):
    # Builds the S-matrix; the linear step response-model
    #
    # Inputs:
    # - Sijs: each SISO vector, ex.: Sijs = [S11, S12, S21, S22] for a 2x2 system
    #         NB! This order is important for the matrix comprehension in building
    #         Si beneath! Shape of Sij: P x 1. Here assumed that P == N
    # - 
    #
    # Returns: S-matrix, shape: n_CV * Hp x n_MV * (N - Hu)
    
    N = Sijs[0].shape[0] # Length of SISO-model
    S = np.zeros((n_CV * N, n_MV * (N - Hu)))
    Sis = np.zeros((n_CV * N, n_MV))

    for i in range(N):

        Si = np.zeros((n_CV * n_MV, 1)) # Will reshape to n_CV * n_MV after
        for index, Sij in enumerate(Sijs):
            Si[index] = Sij[i]
        
        Sis[i * n_CV:(i + 1) * n_CV, :] = np.reshape(Si, (n_CV, n_MV))

    pad_sz = n_CV * (N - Hu)
    Sis = np.pad(Sis, ((pad_sz, 0), (0,0)), mode='constant', constant_values=0)

    for i in range(S.shape[1] // n_MV):
        temp = Sis[pad_sz - (i * n_CV):pad_sz + (N - i) * n_CV, :]
        S[:, i * n_MV:(i + 1) * n_MV] = temp

    return S


def S_xx_append(P, S_xx):
    S_xx_steady = S_xx[(len(S_xx)-1)]
    
    for i in range(P-len(S_xx)):
        S_xx = np.append(S_xx, S_xx_steady)

    return S_xx

def read_S(rel_S_paths):
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

##################################################################################

# ----- VARIABLES AND MATRICES SET-UP ----- #
#### ----- Read config-file ----- ####

config_path = Path(__file__).parent / "../config/mpc_config.yaml"
configs = read_yaml(config_path)

#### ----- Globals - fetch all of these from config-file ----- ####
n_MV = configs['SYSTEM_PARAMETERS']['n_MV']         # number of input variables - given from system (p.426)
n_CV = configs['SYSTEM_PARAMETERS']['n_CV']         # number of outputs - given from system (p. 423) 
Hu = configs['TUNING_PARAMETERS']['Hu']             # control horizon - design parameter (p. 416)
Hp = configs['TUNING_PARAMETERS']['Hp']             # prediction horizon - design parameter (p. 417)
Hw = configs['TUNING_PARAMETERS']['Hw']             # time step from which prediction horizon effectively starts

P_bar = configs['TUNING_PARAMETERS']['P_bar']       # Weight for actuations 
Q_bar = configs['TUNING_PARAMETERS']['Q_bar']       # Weight for errors from ref
n_eh = n_CV                                         # Same as "ny_over_bar"; amount of CVs that have upper limits
n_el = n_CV                                         # Same as "ny_under_bar"; amount of CVs that have lower limits
rho_h_parameter = configs['TUNING_PARAMETERS']['rho_h'], (1,1)
rho_l_parameter = configs['TUNING_PARAMETERS']['rho_l'], (1,1)
rho_h = np.vstack((rho_h_parameter[0][0], rho_h_parameter[0][1])) # Weight for slack variable eps_h
rho_l = np.vstack((rho_l_parameter[0][0], rho_l_parameter[0][1])) # Weight for slack variable eps_l

z_bar_lb = configs['TUNING_PARAMETERS']['z_bar_lb']
z_bar_ub = configs['TUNING_PARAMETERS']['z_bar_ub'] # shape

y_over_bar = configs['TUNING_PARAMETERS']['y_over_bar']        # Upper limit of CVs
y_under_bar = configs['TUNING_PARAMETERS']['y_under_bar']      # Lower limit of CVs
u_over_bar = configs['TUNING_PARAMETERS']['u_over_bar']        # Upper limit of MVs
u_under_bar = configs['TUNING_PARAMETERS']['u_under_bar']      # Lower limit of MVs


rel_S_paths = {'S11': '../data/S11_data_new.npy', 'S21': '../data/S12_data_new.npy',
               'S12': '../data/S21_data_new.npy', 'S22': '../data/S22_data_new.npy'}

# S_11 = np.genfromtxt('data/S_11_gas_choke')
# S_21 = np.genfromtxt('data/S_21_oil_choke')
# S_12 = np.genfromtxt('data/S_12_gas_gaslift')
# S_22 = np.genfromtxt('data/S_22_oil_gaslift')
# S_11 = np.genfromtxt('data/S_11_gas_choke')
# S_21 = np.genfromtxt('data/S_21_oil_choke')
# S_12 = np.genfromtxt('data/S_12_gas_gaslift')
# S_22 = np.genfromtxt('data/S_22_oil_gaslift')
# Sijs = np.vstack((S_11, S_12, S_21, S_22))

Sijs = read_S(rel_S_paths)

N = 100
Sijs_enlarged = np.array([S_xx_append(Hp, siso) for siso in Sijs]) #puttet N*2 for å få den veldig lang til bruk for å finne psi_ij

#### ----- Build Theta, Psi, Upsilon ----- ####

Theta = matrix_generation.get_Theta(Sijs_enlarged, Hp, Hu, Hw)

Psi = matrix_generation.get_Psi(Sijs_enlarged, Hp, Hu, Hw, N)

Upsilon = matrix_generation.get_Upsilon(Sijs_enlarged, Hp, Hw)

##########################################################


#### ----- Build minimization problem matrix Hd (vector gd is variable) - (4.10a) in Kufoalor ----- ####
#! Here assumed that Q == Q_y
Q_y = matrix_generation.get_Q(Q_bar, Hp, Hw)              # shape: (n_CV * (Hp - Hw + 1)) x (n_CV * (Hp - Hw + 1)).
P = matrix_generation.get_P(P_bar, Hu)                    # shape: (n_MV * Hu) x (n_MV * Hu).

Hd_bar = Theta.T @ Q_y @ Theta + P                              # shape: (n_MV * Hu)x(n_MV * Hu)

Hd = np.zeros(((n_MV * Hu) + n_eh + n_el, (n_MV * Hu) + n_eh + n_el))
Hd[:(n_MV * Hu), :(n_MV * Hu)] = Hd_bar
Hd[(n_MV * Hu):(n_MV * Hu) + n_eh, (n_MV * Hu):(n_MV * Hu) + n_eh] = np.zeros((n_eh, n_eh))
Hd[(n_MV * Hu) + n_eh:, (n_MV * Hu) + n_eh:] = np.zeros((n_el, n_el))


gamma = -2 * (Theta.T @ Q_y)

# Instantiate for later
gd = np.zeros((Hd.shape[1], 1))

##########################################################

#### ----- Build inequality system matrix Ad (vector bd is variable) - (4.10d) in Kufoalor ----- ##


G = matrix_generation.get_G(Hp, Hw, n_CV)
F = matrix_generation.get_F(Hu, n_MV)
K_inv = np.linalg.inv(matrix_generation.get_K(Hu, n_MV))
Mh = matrix_generation.get_Mh(Hp, Hw, n_eh)
Ml = matrix_generation.get_Ml(Hp, Hw, n_el)

# --- Assemble Ad --- #
Ad = np.zeros((G.shape[0] + F.shape[0], Theta.shape[1] + Mh.shape[1] + Ml.shape[1]))
vec1 = np.hstack((G@Theta, -Mh, -Ml))
temp1 = F@K_inv
rows = temp1.shape[0]
zeros22 = np.zeros((rows, Mh.shape[1]))
zeros23 = np.zeros((rows, Ml.shape[1]))
vec2 = np.hstack((F@K_inv, zeros22, zeros23))

Ad = np.vstack((vec1, vec2))

##########################################################

# Needed for bd, but constant:
g = matrix_generation.get_g(Hp, Hw, n_CV, y_over_bar, y_under_bar)
f = matrix_generation.get_f(Hu, n_MV, u_over_bar, u_under_bar)
Gamma = matrix_generation.get_Gamma(Hu, n_MV)
FKinvGamma = F @ K_inv @ Gamma

# Instantiate for later
bd = np.zeros((g.shape[0] + f.shape[0], 1))

#### ----- Get desired set-points from file ----- ####
y_ref = configs['SET_POINTS']['ref']          # For now: one constant value per CV fetched from file

#### ----- Initialize inputs ----- ####
U_tilde_prev = np.zeros((n_MV, 1))            # U_tilde(k - 1), p.36 in Kufoalor
U_tilde_N = np.zeros((n_MV, 1))               # U_tilde(k - Nj), where Nj == N
dU_tilde_prev = np.zeros((N-Hw-1, n_MV))
u_prev = np.zeros((n_MV, 1))
U = np.zeros((Hu*n_MV, 1))
U_next = np.zeros((2,1))

#### ----- Initialize outputs ----- ####
y_hat = np.zeros((n_CV, 1))                   # TODO: If we start with a measurement, initialize as that measurement instead
y_hat_k_minus_1 = np.zeros((2,1))
y_prev = np.zeros((n_CV, 1))
y0 = np.zeros((2,1))

#### ----- Initialize bias ----- ####
V = np.zeros((Hp - Hw + 1, 1))

#### ----- Set up time-keeping ----- ####
time = 0
start_time = time
delta_t = configs['RUNNING_PARAMETERS']['delta_t']
final_time = configs['RUNNING_PARAMETERS']['final_time']

#### ----- Create model and run for a given time period - used for "measurements" ----- ####
model_path = Path(__file__).parent / "../fmu/fmu_endret_deadband.fmu"
model, u_sim, y_sim = simulate_fmu.init_model(model_path, start_time, final_time, delta_t, Hp)

y_prev[0] = y_sim[-1][0]
y_prev[1] = y_sim[-1][1]
y_hat[0] = 0
y_hat[1] = 0

y_hat_k_minus_1[0] =y_sim[-1][0]
y_hat_k_minus_1[1] =y_sim[-1][1]

u1_start = u_sim[-1][0] 
u2_start = u_sim[-1][1]
U_tilde_prev[0] = u_sim[-1][0] 
U_tilde_prev[1] = u_sim[-1][1] 

#Get dU_tilde_prev for the given initialization period
u_sim_flip = np.flip(u_sim)
for i in range(N-Hw-1):
    #Flipper u_sim slik at den forrige implementerte inputen er først i u_sim_flip: dU_tilde = [u(k-1), u(k-2), ....]
    dU_tilde_prev[i][0] = u_sim_flip[i+1][0] - u_sim_flip[i][0]
    dU_tilde_prev[i][1] = u_sim_flip[i+1][1] - u_sim_flip[i][1]

dU_tilde_prev = np.hstack((dU_tilde_prev[:,0], dU_tilde_prev[:,1]))
dU_tilde_prev = np.reshape(dU_tilde_prev, (len(dU_tilde_prev), 1))

####### Init data vectors ##########
gas_rate_per_hr_vec = []
oil_rate_per_hr_vec = []
gas_rate_ref_vec = []
oil_rate_ref_vec = []
choke_input = []
gas_lift_input = []
choke_actual = []
gas_lift_actual = []
bias_gas = []
bias_oil = []
t = []

##########
run = True
run_number = 0
##########

y0[0] = y_sim[-1][0]
y0[1] = y_sim[-1][1]

y_hat_init = Psi @ dU_tilde_prev + Upsilon @ U_tilde_prev
y_hat_k_minus_1[0] = y_hat_init[0]
y_hat_k_minus_1[1] = y_hat_init[Hp-Hw+1]




###########----- OPTIMIZATION LOOP ----- ###########
while time < final_time:

    #Get V
    V = matrix_generation.get_V_matrix(y_prev, y_hat_k_minus_1, Hp, Hw)

    #Get Lambda_d
    Lambda_d = Psi @ dU_tilde_prev + Upsilon @ U_tilde_prev + V
    
    #Get T
    T = matrix_generation.get_T(y_ref, n_CV, Hp, Hw)

    # #Do steps in T - refrence trajectory
    # if time > 3000:
    #     if time < 5000:
    #         T[Hp:] = T[Hp:] + 10
    #     else:
    #         T[Hp:] = T[Hp:] + 10
            #T[0:Hp] = T[0:Hp] + 1000
            
    if ((time > 3000) and (time <= 5000)):
        T[Hp:] = T[Hp:] + 10

    if ((time > 5000) and (time <= 7000)):
        T[Hp:] = T[Hp:] + 20

    if ((time > 7000) and (time <= 9000)):
        T[Hp:] = T[Hp:] + 10

    if ((time > 9000) and (time <= 11000)):
        T[Hp:] = T[Hp:] + 10
        T[0:Hp] = T[0:Hp] + 1000

    if ((time > 11000) and (time <= 13000)):
        T[0:Hp] = T[0:Hp] + 1000

    if ((time > 13000) and (time <= 16000)):
        T[Hp:] = T[Hp:] - 30
        T[0:Hp] = T[0:Hp]

    if time > 16000:
        T[Hp:] = T[Hp:] - 30
        T[0:Hp] = T[0:Hp] - 5000


    # if time > final_time/3:
    #     #T[0:Hp] = T[0:Hp]
    #     if time < final_time*2/3:
    #         T[Hp:] = T[Hp:] + 10
    #     else:
    #         T[0:Hp] = T[0:Hp] + 500
    #         T[Hp:] = T[Hp:] + 10

    #Calculate zeta
    zeta = T - Lambda_d

    #Get gd
    gd = np.vstack(((zeta.T @ gamma.T).T, rho_h, rho_l)) # Note! This is transposed in qpsolvers-doc, but not in Kufoalor 
    gd_new = gd[:,0]

    #Get bd
    vec1 = -G@Lambda_d + g
    vec2 = -F@K_inv@Gamma@U_tilde_prev + f
    bd = np.vstack((vec1, vec2))

    # --- Gurobi setup --- #
    m = gp.Model("qp")
    m.Params.LogToConsole = 0

    variables = []
    for i in range(Hu):
        #print(i)
        variables.append(m.addVar(lb=-0.55, ub=0.55, name="dU_1"))

    for i in range(Hu):
        #print(i)
        variables.append(m.addVar(lb=-166.7, ub=166.7, name="dU_2"))


    for i in range(4):
        #print(i)
        variables.append(m.addVar(lb=0.0, ub=10000000, name="slack_var"))

    Constraints = m.addMConstr(Ad, variables, "<=", bd)

    m.setMObjective(Hd, gd_new, 0, None, None, None, GRB.MINIMIZE)

    
    m.update()
    m.optimize()
    #print(m.status)
    #######################################################################

    # --- Extract values from optimization problem --- #
    var_value = []
    for v in m.getVars():
        #print('%s %g' % (v.VarName, v.X))
        var_value.append(v.X)

    dU_1 = var_value[0:Hu]
    dU_2 = var_value[Hu:Hu*2]  #stod egt [20:40]
    dU = np.hstack((dU_1, dU_2))
    dU = np.reshape(dU, (len(dU), 1))

    ########################################################################

    # --- Calculate the input and the predicted output --- #
    U = K_inv@(Gamma@U_tilde_prev + dU)
    U[0:Hu] = U[0:Hu]
    U[Hu:Hu*2] = U[Hu:Hu*2]
    
    y_hat = Psi @ dU_tilde_prev + Upsilon @ U_tilde_prev + Theta@dU

    U_next[0] = U[0]
    U_next[1] = U[Hu]

    ########################################################################

    # --- Do step with the first optimized input --- #
    y_prev[0], y_prev[1], u_prev[0], u_prev[1], choke_opening, gas_lift_current,  = simulate_fmu.simulate_singlewell_step(model, time, delta_t, U_next) # measurement from FMU, i.e. result from previous actuation
    
    # --- Roll the input, and set the newest input at the end --- #
    u_sim = np.roll(u_sim, [-1, -1]) #putter den nyeste inputen bakerst
    u_sim[-1, 0] = u_prev[0]
    u_sim[-1, 1] = u_prev[1]

    U_tilde_prev[0] = u_prev[0]
    U_tilde_prev[1] = u_prev[1]

    # --- Roll the change input, and set the newest change in input at the end --- #
    dU_tilde_prev = np.roll(dU_tilde_prev, 1)
    dU_tilde_prev[0] = u_sim[-1][0] - u_sim[-2][0]
    dU_tilde_prev[N-Hw-1] = u_sim[-1][1] - u_sim[-2][1]

    # --- Collect the first predited output --- #
    y_hat_k_minus_1[0] = y_hat[0]
    y_hat_k_minus_1[1] = y_hat[Hp-Hw+1]

    # --- Set the newest measurement at the end --- #
    y_sim = np.roll(y_sim, [-1, -1]) #putter den nyeste målingen bakerst
    y_sim[-1, 0] = y_prev[0]
    y_sim[-1, 1] = y_prev[1]

    #Plots
    plotte_tid_neg = np.arange(-200, delta_t*(Hp), delta_t)
    plotte_tid = np.arange(0, delta_t*(Hp), delta_t)


    predictions_for_plots = Lambda_d + Theta@dU

    gas_rate_predicted = predictions_for_plots[0:Hp]
    oil_rate_predicted = predictions_for_plots[Hp:]

    ones_vec = np.ones((20,1))
    steady_vec_gas = ones_vec*gas_rate_predicted[0]
    steady_vec_oil = ones_vec*oil_rate_predicted[0]
    steady_u1 = ones_vec*u_prev[0]
    steady_u2 = ones_vec*u_prev[1]

    u1 = U[0:Hu]
    u2 = U[Hu:]

    Neg_values_plot = True
    if Neg_values_plot == True:
        gas_rate_predicted = np.vstack((steady_vec_gas, gas_rate_predicted))
        oil_rate_predicted = np.vstack((steady_vec_oil, oil_rate_predicted))
        u1 = np.vstack((steady_u1, u1))
        u2 = np.vstack((steady_u2, u2))

    gas_rate_ref = T[0:Hp]
    oil_rate_ref = T[Hp:]



    

    gas_rate_per_hr_vec = np.append(gas_rate_per_hr_vec, y_prev[0])
    oil_rate_per_hr_vec = np.append(oil_rate_per_hr_vec, y_prev[1])
    gas_rate_ref_vec = np.append(gas_rate_ref_vec, gas_rate_ref[0])
    oil_rate_ref_vec = np.append(oil_rate_ref_vec, oil_rate_ref[0])


    choke_input = np.append(choke_input, u_prev[0])
    gas_lift_input = np.append(gas_lift_input, u_prev[1])
    choke_actual = np.append(choke_actual, choke_opening)
    gas_lift_actual = np.append(gas_lift_actual, gas_lift_current*1000/24)
    bias_gas = np.append(bias_gas, V[0])
    bias_oil = np.append(bias_oil, V[-1])


    for n in range(Hp-Hu):
        u1 = np.append(u1, U[Hu-1])
        u2 = np.append(u2, U[-1])
    t = np.append(t, time)

    ### --- PLOTTING --- ###
    
    Plot_Prediction = False
    if Plot_Prediction == True:
        figure = plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(plotte_tid_neg, gas_rate_predicted, label = 'Gas Rate Predicted', linewidth = 4)
        plt.plot(plotte_tid, gas_rate_ref, label = 'Reference signal',linewidth = 4, linestyle = "dashed")
        plt.title('Predicted Gas Rate', fontsize=16)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
        plt.ylabel('Gas rate [m3/hr]', fontdict=None, labelpad=None, fontsize=14)
        plt.legend(fontsize = 'x-large')
        #k = 0
        plt.xticks([0, 100, 200, 300, 500, 1000, 1500], ['k', 'k+10', 'k+20', 'k+30', 'k+50', 'k+100', 'k+150'])

        plt.subplot(2,1,2)
        plt.plot(plotte_tid_neg, oil_rate_predicted, label = 'Oil Rate Predicted', linewidth = 4)
        plt.plot(plotte_tid, oil_rate_ref, label = 'Reference signal', linewidth = 4, linestyle = "dashed")
        plt.title('Predicted Oil Rate', fontsize=16)
        plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
        plt.ylabel('Oil rate [m3/hr]', fontdict=None, labelpad=None, fontsize=14)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large')
        plt.xticks([0, 100, 200, 300, 500, 1000, 1500], ['k', 'k+10', 'k+20', 'k+30', 'k+50', 'k+100', 'k+150'])

    Plot_INPUTS = False
    if Plot_INPUTS == True:
        figure = plt.figure(2)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(plotte_tid_neg, u1, label = 'Future: Choke opening [%]', linewidth = 4)
        plt.xticks([0, 100, 200, 300, 500, 1000, 1500], ['k', 'k+10', 'k+20', 'k+30', 'k+50', 'k+100', 'k+150'])
        plt.title('Future choke input', fontsize=16)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large', loc='lower right')

        plt.subplot(2,1,2)
        plt.plot(plotte_tid_neg, u2, label = 'Future: Gas-lift rate [m3/hr]', linewidth = 4)
        plt.title('Future gas-lift input', fontsize=16)
        plt.xticks([0, 100, 200, 300, 500, 1000, 1500], ['k', 'k+10', 'k+20', 'k+30', 'k+50', 'k+100', 'k+150'])
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large', loc='lower right')

    Plot_Prediction_long_horizon = False
    if Plot_Prediction_long_horizon == True:
        figure = plt.figure(1)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(plotte_tid_neg, gas_rate_predicted, label = 'Gas Rate Predicted', linewidth = 4)
        plt.plot(plotte_tid, gas_rate_ref, label = 'Reference signal',linewidth = 4, linestyle = "dashed")
        plt.title('Predicted Gas Rate', fontsize=16)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
        plt.ylabel('Gas rate [m3/hr]', fontdict=None, labelpad=None, fontsize=14)
        plt.legend(fontsize = 'x-large')
        #k = 0
        plt.xticks([0, 300, 1000, 3000, 5000, 7000], ['k', 'k+30', 'k+100', 'k+300', 'k+500', 'k+700'])

        plt.subplot(2,1,2)
        plt.plot(plotte_tid_neg, oil_rate_predicted, label = 'Oil Rate Predicted', linewidth = 4)
        plt.plot(plotte_tid, oil_rate_ref, label = 'Reference signal', linewidth = 4, linestyle = "dashed")
        plt.title('Predicted Oil Rate', fontsize=16)
        plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
        plt.ylabel('Oil rate [m3/hr]', fontdict=None, labelpad=None, fontsize=14)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large')
        plt.xticks([0, 300, 1000, 3000, 5000, 7000], ['k', 'k+30', 'k+100', 'k+300', 'k+500', 'k+700'])

    Plot_INPUTS_long_horizon = False
    if Plot_INPUTS_long_horizon == True:
        figure = plt.figure(2)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(plotte_tid_neg, u1, label = 'Choke opening [%]', linewidth = 4)
        plt.xticks([0, 300, 1000, 3000, 5000, 7000], ['k', 'k+30', 'k+100', 'k+300', 'k+500', 'k+700'])
        plt.title('Choke input', fontsize=16)
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large', loc='lower right')

        plt.subplot(2,1,2)
        plt.plot(plotte_tid_neg, u2, label = 'Gas-lift rate [m3/hr]', linewidth = 4)
        plt.title('Gas-lift input', fontsize=16)
        plt.xticks([0, 300, 1000, 3000, 5000, 7000], ['k', 'k+30', 'k+100', 'k+300', 'k+500', 'k+700'])
        plt.grid(linestyle='dotted', linewidth=1)
        plt.legend(fontsize = 'x-large', loc='lower right')



    # Plot_INPUTS = True
    # if Plot_INPUTS == True:
    #     figure = plt.figure(2)
    #     plt.clf()
    #     plt.subplot(2,2,1)
    #     plt.plot(plotte_tid[0:Hu], U[0:Hu], label = 'U1')
    #     #plt.plot(plotte_tid, gas_rate_ref, label = 'Reference')
    #     plt.title('U1')
    #     plt.subplot(2,2,2)
    #     plt.plot(plotte_tid[0:Hu], U[Hu:], label = 'U2')
    #     #plt.plot(plotte_tid, gas_rate_ref, label = 'Reference')
    #     plt.title('U2')
    #     plt.subplot(2,2,3)
    #     plt.plot(plotte_tid[0:Hu], dU_1, label = 'DeltaU1')
    #     #plt.plot(plotte_tid, gas_rate_ref, label = 'Reference')
    #     plt.title('DeltaU1')       
    #     plt.subplot(2,2,4)
    #     plt.plot(plotte_tid[0:Hu], dU_2, label = 'DeltaU2')
    #     #plt.plot(plotte_tid, gas_rate_ref, label = 'Reference')
    #     plt.title('DeltaU2')       
    #     plt.legend()

    Plot_measurements = False
    if Plot_measurements == True:
        figure = plt.figure(3)
        plt.clf()
        plt.subplot(211)
        plt.plot(t, gas_rate_per_hr_vec, label = 'Gas rate')
        plt.title('Gas rate measurements')
        plt.subplot(212)
        plt.plot(t, oil_rate_per_hr_vec, label = 'Oil rate')
        plt.title('Oil rate measurements')
        plt.legend()

    plt.show()


    # print("y_hat: ", y_hat)
    # print("y_prev: ", y_prev)
    #Må huske å rolle noe y_prev osv, bruke y_prev og u_prev til å rolle u_sim, y_sim og putte inn de nyeste verdiene til y_prev og u_prev
    time += delta_t
    m.dispose() 
    #print(m.getConstrs())
    run_number += 1
    #print(m.getConstrs())
    #print("Optimized dU1: ", dU_1[0])
    #print("Optimized dU2: ", dU_2[0])

    if np.remainder(run_number, delta_t) == 0:
        print("\nRun number: ", run_number, "/", int(final_time/delta_t))

    PRINT = False
    if PRINT == True:
        if np.remainder(run_number, delta_t) == 0:
            print("\nChoke input", u_prev[0])
            print("Gas-lift input:", u_prev[1])

            print("\nGas error: ", V[0])
            print("Oil error: ", V[-1])

            print("\nPredicted gasrate: ", y_hat[0])
            print("Actual gas rate: ", y_prev[0])
            
            print("\nPredicted oilrate: ", y_hat[Hp-Hw+1])
            print("Actual oil rate: ", y_prev[1])
            
        
    #run = False


np.save('t.npy', t)
np.save('oil_rate_per_hr_vec.npy', oil_rate_per_hr_vec)
np.save('oil_rate_ref_vec.npy', oil_rate_ref_vec)

np.save('gas_rate_per_hr_vec.npy', gas_rate_per_hr_vec )
np.save('gas_rate_ref_vec.npy', gas_rate_ref_vec)

np.save('choke_input.npy', choke_input)
np.save('gas_lift_input.npy', gas_lift_input)
np.save('choke_actual.npy', choke_actual)

np.save('bias_gas.npy', bias_gas)
np.save('bias_oil.npy', bias_oil)

print("Saved data")

PLOT = False
if PLOT == True:
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
    plt.plot(t, choke_input, label = 'Choke input', linewidth = 2)
    plt.plot(t, choke_actual, label = 'Actual choke opening', linewidth = 2)
    plt.title('Choke input', fontsize=16)
    plt.grid(linestyle='dotted', linewidth=1)
    plt.xlabel('Time', fontdict=None, labelpad=None, fontsize=14)
    plt.ylabel('Choke input [%]', fontdict=None, labelpad=None, fontsize=14)
    plt.legend(fontsize = 'x-large', loc='lower right')




    plt.subplot(212)
    plt.plot(t, gas_lift_input, label = 'Gas-lift rate', linewidth = 2)
    #plt.plot(t, gas_lift_actual, label = 'Actual gas-lift rate', linewidth = 0.5)
    plt.title('Gas-lift rate', fontsize=16)
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

print("Dear son. If you're seeing this, that means that I, the script am dead - or finished, at least")
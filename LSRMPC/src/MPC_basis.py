import numpy as np
from qpsolvers import solve_qp
import matrix_generation
from yaml import safe_load
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import simulate_fmu

# ----- FUNCTIONS ----- #

def build_V_matrix(y_measure, y_hat_prev, Hp, Hw):
    v1 = y_measure[0] - y_hat_prev[0]
    v2 = y_measure[1] - y_hat_prev[1]
    v1_vec = []
    v2_vec = []
    for i in range(Hp-Hw+1):
        v1_vec = np.append(v1_vec, v1)
        v2_vec = np.append(v2_vec, v2)
    V = np.hstack((v1_vec, v2_vec))
    V = np.reshape(V, ((Hp-Hw+1)*2, 1))
    return V

def build_Y0_matrix(y_measure, Hp, Hw):
    y0_1 = y_measure[0]
    y0_2 = y_measure[1]
    y0_1_vec = []
    y0_2_vec = []
    for i in range(Hp-Hw+1):
        y0_1_vec = np.append(y0_1_vec, y0_1)
        y0_2_vec = np.append(y0_2_vec, y0_2)
    Y0 = np.hstack((y0_1_vec, y0_2_vec))
    Y0 = np.reshape(Y0, ((Hp-Hw+1)*2, 1))
    return Y0

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

def build_full_S(Sijs, n_CV, n_MV, N):
    # Builds the S-matrix; the linear step response-model
    #
    # Inputs:
    # - Sijs: each SISO vector, ex.: Sijs = [S11, S12, S21, S22] for a 2x2 system
    #         NB! This order is important for the matrix comprehension in building
    #         Si beneath! Shape of Sij: P x 1. Here assumed that P == N
    # - 
    #
    # Returns: S-matrix, shape: n_CV * Hp x n_MV * (N - Hu)
    
    S = np.zeros((n_CV * N, n_MV))
    for n, Sij in enumerate(Sijs):
        S[(n // n_MV) * N:((n // n_MV) + 1) * N, n % n_MV] = Sij

    return S

def S_xx_append(P, S_xx):
    S_xx_steady = S_xx[(len(S_xx) - 1)]
    
    for i in range(P - len(S_xx)):
        S_xx = np.append(S_xx, S_xx_steady)

    return S_xx

def get_theta_ij(s_ij, Hp, Hu, Hw):
    numRows = Hp-Hw+1 # Number of rows in resulting matrix
    numCols = Hu # Number of columns in resulting matrix

    res = np.zeros((numRows, numCols))

    startIdx = 0
    endIdx = Hw
    for i in range(numRows):
        arr = s_ij[startIdx:endIdx]
        arr = np.flip(arr)
        res[i][0:len(arr)] = arr
        endIdx += 1
        if (i >Hu-Hw -1):
            startIdx += 1
    return np.array(res)

def get_psi_ij(s_ij_enlarged, Hp, Hu, Hw, N):
    numRows = Hp-Hw+1 # Number of rows in resulting matrix
    numCols = N-Hw-1  # Number of columns in resulting matrix


    res = np.zeros((numRows, numCols))

    startIdx = Hw
    endIdx = N-1
    for i in range(numRows):
        arr = s_ij_enlarged[startIdx:endIdx]
        res[i][0:numCols] = arr
        if endIdx < len(s_ij_enlarged):
            startIdx += 1
            endIdx += 1
        else:
            startIdx = startIdx
            endIdx = endIdx

    return np.array(res)

def get_Upsilon_ij(s_ij_enlarged, Hp, Hw):
    last_element = s_ij_enlarged[-1]
    res = []
    for i in range(Hp-Hw+1):
        res.append(last_element)

    return res

def insert_arr(a, b):
    # Inserts arr b into arr a

    for i in range(len(a)):
        a[i] = b[i]
    
    return a

# ----- VARIABLES AND MATRICES SET-UP ----- #
DEBUG = False
#### ----- Read config-file ----- ####

config_path = Path(__file__).parent / "../config/mpc_config.yaml"
configs = read_yaml(config_path)

#### ----- Globals - fetch all of these from config-file? ----- ####
n_MV = configs['SYSTEM_PARAMETERS']['n_MV'] # number of input variables - given from system (p.426)
n_CV = configs['SYSTEM_PARAMETERS']['n_CV'] # number of outputs - given from system (p. 423) 
Hu = configs['TUNING_PARAMETERS']['Hu'] # control horizon - design parameter (p. 416)
Hp = configs['TUNING_PARAMETERS']['Hp'] # prediction horizon - design parameter (p. 417)
Hw = configs['TUNING_PARAMETERS']['Hw'] # time step from which prediction horizon effectively starts

P_bar = configs['TUNING_PARAMETERS']['P_bar'] # Weight for actuations 
Q_bar = configs['TUNING_PARAMETERS']['Q_bar'] # Weight for errors from ref
n_eh = n_CV     # Same as "ny_over_bar"; amount of CVs that have upper limits
n_el = n_CV     # Same as "ny_under_bar"; amount of CVs that have lower limits
rho_h = np.ones((n_eh, 1)) # Essentially choosing to include both slack variables in opt.prob. (ref. (4.4) in Kufoalor)
rho_l = np.ones((n_el, 1)) # Essentially choosing to include both slack variables in opt.prob. (ref. (4.4) in Kufoalor)
z_bar_lb = configs['TUNING_PARAMETERS']['z_bar_lb']
z_bar_ub = configs['TUNING_PARAMETERS']['z_bar_ub'] # shape: TODO

y_over_bar = configs['TUNING_PARAMETERS']['y_over_bar']        # Upper limit of CVs
y_under_bar = configs['TUNING_PARAMETERS']['y_under_bar']      # Lower limit of CVs
u_over_bar = configs['TUNING_PARAMETERS']['u_over_bar']          # Upper limit of MVs
u_under_bar = configs['TUNING_PARAMETERS']['u_under_bar']         # Lower limit of MVs

S_rel_path = Path(__file__).parent / "../data"
S_11 = np.genfromtxt(S_rel_path / "S_11_gas_choke")
S_12 = np.genfromtxt(S_rel_path / "S_12_gas_gaslift")
S_21 = np.genfromtxt(S_rel_path / "S_21_oil_choke")
S_22 = np.genfromtxt(S_rel_path / "S_22_oil_gaslift")

Sijs = np.vstack((S_11, S_12, S_21, S_22))
N = np.max([len(s) for s in Sijs])
Sijs_enlarged = np.array([S_xx_append(N, siso) for siso in Sijs])
# S = build_full_S(Sijs_enlarged, n_CV, n_MV, N) # Kept if useful when refactored get_Theta/Psi/Upsilon

#### ----- Theta ----- ####
theta_11 = get_theta_ij(Sijs_enlarged[0], Hp, Hu, Hw)
theta_12 = get_theta_ij(Sijs_enlarged[1], Hp, Hu, Hw)
theta_21 = get_theta_ij(Sijs_enlarged[2], Hp, Hu, Hw)
theta_22 = get_theta_ij(Sijs_enlarged[3], Hp, Hu, Hw)

theta1 = np.hstack((theta_11, theta_12))
theta2 = np.hstack((theta_21, theta_22))

Theta = np.vstack((theta1, theta2))

#### ----- Psi ----- ####
psi_11 = get_psi_ij(Sijs_enlarged[0], Hp, Hu, Hw, N)
psi_12 = get_psi_ij(Sijs_enlarged[1], Hp, Hu, Hw, N)
psi_21 = get_psi_ij(Sijs_enlarged[2], Hp, Hu, Hw, N)
psi_22 = get_psi_ij(Sijs_enlarged[3], Hp, Hu, Hw, N)

psi1 = np.hstack((psi_11, psi_12))
psi2 = np.hstack((psi_21, psi_22))

Psi = np.vstack((psi1, psi2))

#### ----- Upsilon ----- ####
upsilon_11 = get_Upsilon_ij(Sijs_enlarged[0], Hp, Hw)
upsilon_12 = get_Upsilon_ij(Sijs_enlarged[1], Hp, Hw)
upsilon_21 = get_Upsilon_ij(Sijs_enlarged[2], Hp, Hw)
upsilon_22 = get_Upsilon_ij(Sijs_enlarged[3], Hp, Hw)

upsilon_11 = np.reshape(upsilon_11, (len(upsilon_11), 1))
upsilon_12 = np.reshape(upsilon_12, (len(upsilon_12), 1))
upsilon_21 = np.reshape(upsilon_21, (len(upsilon_21), 1))
upsilon_22 = np.reshape(upsilon_22, (len(upsilon_22), 1))

Upsilon1 = np.hstack((upsilon_11, upsilon_12))
Upsilon2 = np.hstack((upsilon_21, upsilon_22))

Upsilon = np.vstack((Upsilon1, Upsilon2))

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

#### ----- Build inequality system matrix Ad (vector bd is variable) - (4.10d) in Kufoalor ----- ##
G = matrix_generation.get_G(Hp, Hw, n_CV)
F = matrix_generation.get_F(Hu, n_MV)
K_inv = np.linalg.inv(matrix_generation.get_K(Hu, n_MV))
Mh = matrix_generation.get_Mh(Hp, Hw, n_eh)
Ml = matrix_generation.get_Ml(Hp, Hw, n_el)

Ad = np.zeros((G.shape[0] + F.shape[0], Theta.shape[1] + Mh.shape[1] + Ml.shape[1]))
Ad[:G.shape[0], :Theta.shape[1]] = G @ Theta
Ad[:G.shape[0], Theta.shape[1]:Theta.shape[1] + Mh.shape[1]] = -Mh
Ad[:G.shape[0], Theta.shape[1] + Mh.shape[1]:] = -Ml
Ad[G.shape[0]:, :K_inv.shape[1]] = F @ K_inv

# Needed for bd, but constant:
g = matrix_generation.get_g(Hp, Hw, n_CV, y_over_bar, y_under_bar)
f = matrix_generation.get_f(Hu, n_MV, u_over_bar, u_under_bar)
Gamma = matrix_generation.get_Gamma(Hu, n_MV)
FKinvGamma = F @ K_inv @ Gamma

# Instantiate for later
bd = np.zeros((g.shape[0] + f.shape[0], 1))

#### ----- Get desired set-points from file. Initialize U_tilde, dU_tilde and V (disturbance) ----- ####
U_tilde_prev = np.zeros((n_MV, 1))            # U_tilde(k - 1), p.36 in Kufoalor
U_tilde_N = np.zeros((n_MV, 1))               # U_tilde(k - Nj), where Nj == N
dU_tilde_prev = np.zeros((N-Hw-1, n_MV))      # Later reshaped to stacked vectors
u_prev = np.zeros((n_MV, 1))
U_next = np.zeros((2,1))

y_ref = configs['SET_POINTS']['ref'] # For now: one constant value per CV fetched from file
y_hat = np.zeros((n_CV, 1))                  
y_prev = np.zeros((n_CV, 1))

U = np.zeros((Hu*n_MV, 1))
V = np.zeros((Hp - Hw + 1, 1))

#### ----- Set up time-keeping ----- ####
time = 0
start_time = time
delta_t = configs['RUNNING_PARAMETERS']['delta_t']
final_time = configs['RUNNING_PARAMETERS']['final_time']

#### ----- Create model - used for "measurements" ----- ####
model_path = Path(__file__).parent / "../fmu/SingleWell_filtGas.fmu"
model, u_sim, y_sim = simulate_fmu.init_model(model_path, start_time, final_time, delta_t, Hp) #Har lag til at man kjører modellen i 1000sek når man initialiserer, har lag til at man henter ut en u_sim vektor

Y0 = build_Y0_matrix(y_sim[-1], Hp, Hw) # y_sim[-1] is the preivous measurement
# y0 = insert_arr(a=np.zeros((2,1))[:], b=y_sim[-1])
# Y0 = build_Y0_matrix(y0, Hp, Hw)

# Initial guesses should be initial state (current state in simulation)
y_hat = insert_arr(a=y_hat[:], b=y_sim[-1])

y_hat_prev = insert_arr(a=np.zeros((2,1))[:], b=y_sim[-1]) # OMDØPTE y_hat_k_minus_1 TIL y_hat_prev FOR CONSISTENCY MED RESTEN AV KODEN

U_tilde_prev = insert_arr(a=U_tilde_prev[:], b=u_sim[-1])

#### ----- Filling out dU_tilde_prev ----- ####
u_sim_flip = np.flip(u_sim) # Flipped such that previously implemented input goes first: dU_tilde = [u(k-1), u(k-2), ....]

for i in range(N-Hw-1):
    dU_tilde_prev[i] = insert_arr(a=dU_tilde_prev[i,:], b=(u_sim_flip[i+1] - u_sim_flip[i]))

dU_tilde_prev = np.hstack((dU_tilde_prev[:,0], dU_tilde_prev[:,1]))
dU_tilde_prev = np.reshape(dU_tilde_prev, (len(dU_tilde_prev), 1))

#### ----- Some lists to hold values for plotting ----- ####
gas_rate_per_hr_vec = []
oil_rate_per_hr_vec = []
gas_rate_ref_vec = []
oil_rate_ref_vec = []
choke_input = []
gas_lift_input = []
t = []

# ----- OPTIMIZATION LOOP ----- #
itr = 0
Plot_stepwise = False
# Plot_Prediction = False
# Plot_INPUTS = False
Plot_measurements = False
while time < final_time:
    
    #### ----- Updating gd and bd ----- ####
    y_prev = insert_arr(a=y_prev[:], b=y_sim[-1]) # getting the last measurements from the simulation
    V = build_V_matrix(y_prev, y_hat_prev, Hp, Hw) # Generate V

    T = matrix_generation.get_T(y_ref, n_CV, Hp, Hw)
    Lambda_d = Y0 + Psi @ dU_tilde_prev + Upsilon @ U_tilde_prev + V
    zeta = T - Lambda_d

    gd = np.vstack(((zeta.T @ gamma.T).T, rho_h, rho_l))
    gd_new = gd.ravel() # Unrolls gd into a single vector

    bd[:g.shape[0]] = -G @ Lambda_d + g
    bd[g.shape[0]:g.shape[0] + f.shape[0]] = -FKinvGamma @ U_tilde_prev + f

    #### ----- Gurobi ----- ####
    m = gp.Model("qp")
    m.Params.LogToConsole = 0

    # Create variables
    variables = []
    for i in range(Hu):
        variables.append(m.addVar(lb=-0.55, ub=0.55, name="dU_1"))
    for i in range(Hu):
        variables.append(m.addVar(lb=-166.7, ub=166.7, name="dU_2"))
    for i in range(4):
        variables.append(m.addVar(lb=0.0, ub=10000000, name="slack_var"))

    # Define optimization problem
    m.addMConstr(Ad, variables, "<=", bd)
    m.setMObjective(Hd, gd_new, 0, None, None, None, GRB.MINIMIZE)

    # Solve optimization problem
    m.optimize()

    #### ----- Extract solution and run simulation step with optimized actuation ----- ####
    var_value = []
    for v in m.getVars():
        var_value.append(v.X)
        #var_value = [v.X for v in m.getVars()]
    dU = np.hstack((var_value[0:Hu], var_value[Hu:Hu*2]))
    dU = np.reshape(dU, (len(dU), 1))

    U_tilde_prev = insert_arr(a=U_tilde_prev[:], b=u_sim[-1])
    
    U = K_inv @ (Gamma @ U_tilde_prev + dU)
    U_next[0], U_next[1] = U[0], U[Hu]

    y_prev[0], y_prev[1], u_prev[0], u_prev[1] = simulate_fmu.simulate_singlewell_step(model, time, delta_t, U_next) # measurement from FMU, i.e. result from previous actuation

    u_sim = np.roll(u_sim, [-1, -1]) # Make space for new last input
    u_sim[-1] = insert_arr(a=u_sim[-1], b=u_prev) # Put new last input in prepared space
    
    #### ----- Updating before next iteration ----- ####
    y_hat = Lambda_d + Theta @ dU

    #VELDIG USIKKER PÅ OM DETTE ER RIKTIG MÅTE Å IMPLEMENTERE DU_THILDE_PREV PÅ
    dU_tilde_prev = np.roll(dU_tilde_prev, 1)
    dU_tilde_prev[0] = u_sim[-1][0] - u_sim[-2][0]
    dU_tilde_prev[N-Hw-1] = u_sim[-1][1] - u_sim[-2][1] #usikker på om det skal stå dU_tilde_prev[N-Hw - 1] MINUS 1
    
    y_hat_prev[0], y_hat_prev[1] = y_hat[0], y_hat[Hp-Hw+1]

    y_sim = np.roll(y_sim, [-1, -1]) # the newest measurement goes in the back
    y_sim[-1] = insert_arr(a=y_sim[-1], b=y_prev)

    #### ----- Plots for each time step ----- ####
    plotte_tid = np.linspace(0, delta_t*(Hp+1), Hp+1) + time # Offset to match curr. time
    
    gas_rate_predicted = y_hat[0:Hp+1]
    oil_rate_predicted = y_hat[Hp+1:]
    
    gas_rate_ref = T[0:Hp - Hw + 1]
    oil_rate_ref = T[Hp - Hw + 1:]

    gas_rate_per_hr_vec = np.append(gas_rate_per_hr_vec, y_prev[0])
    oil_rate_per_hr_vec = np.append(oil_rate_per_hr_vec, y_prev[1])
    gas_rate_ref_vec = np.append(gas_rate_ref_vec, gas_rate_ref[0])
    oil_rate_ref_vec = np.append(oil_rate_ref_vec, oil_rate_ref[0])

    choke_input = np.append(choke_input, u_prev[0])
    gas_lift_input = np.append(gas_lift_input, u_prev[1])
    
    if Plot_stepwise == True and time == 720:
        fig4, axs3 = plt.subplots(2)
        fig4.suptitle('Predicted future outputs, given calculated inputs', fontsize=23)
        axs3[0].plot(plotte_tid, gas_rate_predicted, label = 'Gas rate predicted')
        axs3[0].plot(plotte_tid, gas_rate_ref, label = 'Gas rate reference')
        # axs3[0].set_title('Gas rate predicted', fontsize=15)
        axs3[0].set_xlabel('time [s]', fontsize=15)
        axs3[0].set_ylabel('Gas flow rate [m^3/h]', fontsize=15)
        axs3[0].legend(loc='center right', prop={'size': 20})
        axs3[0].set_xlim([plotte_tid[0], plotte_tid[-1]])

        axs3[1].plot(plotte_tid, oil_rate_predicted, label = 'Oil rate predicted')
        axs3[1].plot(plotte_tid, oil_rate_ref, label = 'Oil rate reference')
        # axs3[1].set_title('Oil rate predicted', fontsize=15)
        axs3[1].set_xlabel('time [s]', fontsize=15)
        axs3[1].set_ylabel('Oil flow rate [m^3/h]', fontsize=15)
        axs3[1].legend(loc='center right', prop={'size': 20})
        axs3[1].set_xlim([plotte_tid[0], plotte_tid[-1]])

        fig3, axs2 = plt.subplots(2)
        fig3.suptitle('Future inputs given optimal control sequence', fontsize=23)
        axs2[0].plot(plotte_tid[0:Hu], U[0:Hu], 'r', label = 'choke-opening')
        # axs2[0].set_title('Choke valve opening', fontsize=15)
        axs2[0].set_xlabel('time [s]', fontsize=15)
        axs2[0].set_ylabel('choke valve opening [%]', fontsize=15)
        axs2[0].legend(loc='center right', prop={'size': 20})
        
        axs2[1].plot(plotte_tid[0:Hu], U[Hu:], 'r', label = 'gas-lift rate')
        # axs2[1].set_title('Gas-lift rate', fontsize=15)
        axs2[1].set_xlabel('time [s]', fontsize=15)
        axs2[1].set_ylabel('gas-lift rate [m^3/h]', fontsize=15)
        axs2[1].legend(loc='center right', prop={'size': 20})

        fig5, axs4 = plt.subplots(2)
        fig5.suptitle(f'Changes in inputs as calculated by MPC at time {time}', fontsize=23)
        axs4[0].plot(plotte_tid[0:Hu], dU[0:Hu], 'g', label = 'change in choke-opening')
        # axs4[0].set_title('Change in choke-opening', fontsize=15)
        axs4[0].set_xlabel('time [s]', fontsize=15)
        axs4[0].set_ylabel(f'change in choke [%/({delta_t}s)]', fontsize=15)
        axs4[0].legend(loc='center right', prop={'size': 20})

        axs4[1].plot(plotte_tid[0:Hu], dU[Hu:Hu*2], 'g', label = 'change in gas-lift rate')
        # axs4[1].set_title('Change in gas-lift rate', fontsize=15)
        axs4[1].set_xlabel('time [s]', fontsize=15)
        axs4[1].set_ylabel(f'change in gas-lift [m^3/h/({delta_t}s)]', fontsize=15)
        axs4[1].legend(loc='center right', prop={'size': 20})
        # axs20twin = axs2[0].twinx()
        # axs20twin.plot(plotte_tid[0:Hu], dU[0:Hu], 'g--', label = 'change in choke')
        # axs20twin.set_ylabel(f'change in choke [%/({delta_t}s)]', fontsize=15)
        # axs20twin.legend(loc='center right', prop={'size': 15})
        # axs21twin = axs2[1].twinx()
        # axs21twin.plot(plotte_tid[0:Hu], dU[Hu:Hu*2], 'g--', label = 'change in gas-lift')
        # axs21twin.set_ylabel(f'change in gas-lift [m^3/h/({delta_t}s)]', fontsize=15)
        # axs21twin.legend(loc='center right')
        # figure = plt.figure(2)

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

    t = np.append(t, time)
    time += delta_t
    itr += 1
    print(f"iteration {itr}")

# ----- FINAL PLOTS ----- #

fig1, axs0 = plt.subplots(2, sharex=True)

fig1.suptitle('Gas rate and oil rate controlled by MPC', fontsize=23)

axs0[0].plot(t, gas_rate_per_hr_vec, label = 'Gas rate')
axs0[0].plot(t, gas_rate_ref_vec, label = 'Gas rate ref.')
axs0[0].set_title('Gas rate measurements', fontsize=20)
axs0[0].legend(loc='center right', prop={'size': 15})
# axs0[0].set_xlabel('time [s]', fontsize=15)
axs0[0].set_ylabel('Gas flow rate [m^3/h]', fontsize=15)

axs0[1].plot(t, oil_rate_per_hr_vec, label = 'Oil rate',)
axs0[1].plot(t, oil_rate_ref_vec, label = 'Oil rate ref.')
axs0[1].set_title('Oil rate measurements', fontsize=20)
axs0[1].legend(loc='center right', prop={'size': 15})
axs0[1].set_xlabel('time [s]', fontsize=15)
axs0[1].set_ylabel('Oil flow rate [m^3/h]', fontsize=15)

# axs00twin = axs0[0].twinx()
# axs00twin.plot(t, choke_input, 'r--', label = 'Choke')
# axs00twin.legend(loc='lower right', prop={'size': 15})
# axs00twin.set_ylabel('choke valve opening [%]', fontsize=15)
# axs01twin = axs0[1].twinx()
# axs01twin.plot(t, gas_lift_input, 'r--', label = 'Gas-lift rate')
# axs01twin.legend(loc='lower right', prop={'size': 15})
# axs01twin.set_ylabel('gas-lift rate [m^3/h]', fontsize=15)

fig2, axs1 = plt.subplots(2, sharex=True)
fig2.suptitle('Inputs as calculated my MPC', fontsize=23)
axs1[0].plot(t, choke_input, 'r', label = 'Choke')
axs1[0].set_title('Input: choke-opening', fontsize=20)
axs1[0].legend(loc='center right', prop={'size': 15})
# axs1[0].set_xlabel('time [s]', fontsize=15)
axs1[0].set_ylabel('choke valve opening [%]', fontsize=15)
axs1[1].plot(t, gas_lift_input, 'r', label = 'Gas-lift rate')
axs1[1].set_title('Input: gas-lift rate', fontsize=20)
axs1[1].legend(loc='center right', prop={'size': 15})
axs1[1].set_xlabel('time [s]', fontsize=15)
axs1[1].set_ylabel('gas-lift rate [m^3/h]', fontsize=15)


plt.show()

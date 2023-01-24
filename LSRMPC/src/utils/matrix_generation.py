import numpy as np

# ----- FUNCTION DEFINITIONS ----- #

####################################

# -- Q matrix -- #
def get_Q(Q_bar, Hp, Hw):
    # Calculates Q-matrix
    #
    # Input:
    # - Q_bar:  initial weighting values for different y, shape: n_CV x 1
    # - Hp:     prediction horizon
    # - Hw:     where the prediction horizon starts
    # Returns: Q matrix
    Q = np.kron(np.diag(Q_bar), np.eye(Hp - Hw + 1))
    return np.kron(np.diag(Q_bar), np.eye(Hp - Hw + 1))
    
    #return Q-matrix

####################################

# -- P matrix -- #
def get_P(P_bar, Hu):
    # Calculates Pj-matrix
    #
    # Input:
    # - P_bar:  initial weightning value
    # - Hu:     control horizon
    #
    # Returns: P matrix, shape: (n_MV * Hu) x (n_MV * Hu)

    return np.kron(np.diag(P_bar), np.eye(Hu))

####################################

# -- T matrix -- #
def get_T(ref, n_cv, Hp, Hw): # Only for case of constant ref. Not a problem if T is updated each step anyway?
    # Calculates T-matrix
    #
    # Input:
    # - ref:    constant refrence - columnvector (n_cv x 1)
    # - n_cv:   number of outputs
    # - Hp:     prediction horizon
    # - Hw:     where the prediction horizon startsn
    #
    # Returns: T matrix

    temp = np.ones(((Hp-Hw+1), n_cv))
    X = []

    for i in range(len(ref)):
        X = np.append(X, ref[i]*temp[:,i])

    return np.reshape(X, (n_cv * (Hp - Hw + 1), 1))


####################################

# -- Theta matrix -- #
def get_Theta_ij(s_ij, Hp, Hu, Hw):
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

def get_Theta(s_ij, Hp, Hu, Hw):
    theta_11 = get_Theta_ij(s_ij[0], Hp, Hu, Hw)
    theta_12 = get_Theta_ij(s_ij[1], Hp, Hu, Hw)
    theta_21 = get_Theta_ij(s_ij[2], Hp, Hu, Hw)
    theta_22 = get_Theta_ij(s_ij[3], Hp, Hu, Hw)

    theta1 = np.hstack((theta_11, theta_12))
    theta2 = np.hstack((theta_21, theta_22))

    Theta = np.vstack((theta1, theta2))
    return Theta



####################################

# -- Psi matrix -- #
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

def get_Psi(s_ij, Hp, Hu, Hw, N):
    psi_11 = get_psi_ij(s_ij[0], Hp, Hu, Hw, N)
    psi_12 = get_psi_ij(s_ij[1], Hp, Hu, Hw, N)
    psi_21 = get_psi_ij(s_ij[2], Hp, Hu, Hw, N)
    psi_22 = get_psi_ij(s_ij[3], Hp, Hu, Hw, N)

    psi1 = np.hstack((psi_11, psi_12))
    psi2 = np.hstack((psi_21, psi_22))

    Psi = np.vstack((psi1, psi2))

    return Psi
####################################

# -- Upsilon matrix -- #
def get_Upsilon_ij(s_ij_enlarged, Hp, Hw):
    last_element = s_ij_enlarged[-1]
    res = []
    for i in range(Hp-Hw+1):
        res.append(last_element)

    return res

def get_Upsilon(s_ij, Hp, Hw):

    upsilon_11 = get_Upsilon_ij(s_ij[0], Hp, Hw)
    upsilon_12 = get_Upsilon_ij(s_ij[1], Hp, Hw)
    upsilon_21 = get_Upsilon_ij(s_ij[2], Hp, Hw)
    upsilon_22 = get_Upsilon_ij(s_ij[3], Hp, Hw)

    upsilon_11 = np.reshape(upsilon_11, (len(upsilon_11), 1))
    upsilon_12 = np.reshape(upsilon_12, (len(upsilon_12), 1))
    upsilon_21 = np.reshape(upsilon_21, (len(upsilon_21), 1))
    upsilon_22 = np.reshape(upsilon_22, (len(upsilon_22), 1))

    Upsilon1 = np.hstack((upsilon_11, upsilon_12))
    Upsilon2 = np.hstack((upsilon_21, upsilon_22))

    Upsilon = np.vstack((Upsilon1, Upsilon2))

    return Upsilon

####################################

# -- V matrix -- #
def get_V_matrix(y_measure, y_hat_k_minus_1, Hp, Hw):
    v1 = y_measure[0] - y_hat_k_minus_1[0]
    v2 = y_measure[1] - y_hat_k_minus_1[1]
    v1_vec = []
    v2_vec = []
    for i in range(Hp-Hw+1):
        v1_vec = np.append(v1_vec, v1)
        v2_vec = np.append(v2_vec, v2)
    V = np.hstack((v1_vec, v2_vec))
    V = np.reshape(V, ((Hp-Hw+1)*2, 1))
    return V

####################################

# -- Mh and Ml matrix -- #
def get_Mh(Hp, Hw, ny_over_bar):
    # Builds Mh matrix
    #
    # Input:
    # - Hp:             prediction horizon
    # - Hw:             where the prediction horizon starts
    # - ny_over_bar:    amount of upper limited outputs (CVs)
    #
    # Returns: Mh for the whole MIMO model, shape: ((Hp - Hw + 1) * 2 * ny_over_bar) x ny_over_bar

    ml_over_bar = np.kron(np.ones(Hp - Hw + 1), np.array([1, 0]))
    return np.kron(np.eye(ny_over_bar), ml_over_bar).T

def get_Ml(Hp, Hw, ny_under_bar):
    # Builds Mh matrix
    #
    # Input:
    # - Hp:             prediction horizon
    # - Hw:             where the prediction horizon starts
    # - ny_under_bar:   amount of lower limited outputs (CVs)
    #
    # Returns: Mh for the whole MIMO model, shape: ((Hp - Hw + 1) * 2 * ny_under_bar) x ny_under_bar

    ml_under_bar = np.kron(np.ones(Hp - Hw + 1), np.array([0, 1]))
    return np.kron(np.eye(ny_under_bar), ml_under_bar).T

####################################

# -- F and f matrix -- #
def get_F(Hu, n_MV):
    # Builds F matrix (p.35 of Kufoalor)
    #
    # Input:
    # - Hu:    control horizon
    # - n_MV:  amount of inputs (MVs)
    #
    # Returns: F matrix, shape: (Hu * n_MV * 2) x (Hu * n_MV)

    factor = np.array([1, -1])
    identity = np.eye(Hu)
    Fi = np.kron(identity, factor)
    
    return np.kron(np.identity(n_MV), Fi).T

def get_f(Hu, n_MV, u_over_bar, u_under_bar):
    # Builds the inequality constraint f (p. 35 of Kufoalor)
    #
    # Input:
    # - Hu:             control horizon
    # - n_MV:           amount of inputs (MVs)
    # - u_over_bar:     upper limit for input (NB! Static and constant for all MVs)
    # - u_under_bar:    lower limit for input (NB! Static and constant for all MVs)
    #
    # Returns: f vector, shape: (Hu * n_MV * 2) x 1
    
    one = np.ones((Hu,))
    fis = []
    for j in range(n_MV):
        factor = np.array([u_over_bar[j], -u_under_bar[j]])
        fis.append(np.kron(one, factor))
    
    return np.reshape(np.hstack(tuple(fis)), (Hu * n_MV * 2, 1))

####################################

# -- E and e matrix -- #
def get_E(Hu, n_MV):
    # Builds F matrix (p.35 of Kufoalor)
    #
    # Input:
    # - Hu:    control horizon
    # - n_MV:  amount of inputs (MVs)
    #
    # Returns: F matrix, shape: (Hu * n_MV * 2) x (Hu * n_MV)

    factor = np.array([1, -1])
    identity = np.eye(Hu)
    Ei = np.kron(identity, factor)
    
    return np.kron(np.identity(n_MV), Ei).T

def get_e(Hu, n_MV, du_over_bar, du_under_bar):
    # Builds the inequality constraint f (p. 35 of Kufoalor)
    #
    # Input:
    # - Hu:             control horizon
    # - n_MV:           amount of inputs (MVs)
    # - u_over_bar:     upper limit for input (NB! Static and constant for all MVs)
    # - u_under_bar:    lower limit for input (NB! Static and constant for all MVs)
    #
    # Returns: f vector, shape: (Hu * n_MV * 2) x 1
    
    one = np.ones((Hu,))
    eis = []
    for j in range(n_MV):
        factor = np.array([du_over_bar[j], -du_under_bar[j]])
        eis.append(np.kron(one, factor))
    
    return np.reshape(np.hstack(tuple(eis)), (Hu * n_MV * 2, 1))


####################################

# -- G and g matrix -- #
def get_G(Hp, Hw, n_CV):
    # Builds G matrix (p.35 of Kufoalor)
    #
    # Input:
    # - Hp:    prediction horizon
    # - Hw:    where the prediction horizon starts
    # - n_MV:  amount of outputs (CVs)
    #
    # Returns: G matrix, shape: ((Hp - Hw + 1) * (nyoverbar + nyunderbar)) x ((Hp - Hw + 1) * n_CV)

    factor = np.array([1, -1])
    identity = np.eye(Hp - Hw + 1)
    Gi = np.kron(identity, factor)

    return np.kron(np.identity(n_CV), Gi).T

def get_g(Hp, Hw, n_CV, y_over_bar, y_under_bar):
    # Builds the inequality constraint g (p. 35 of Kufoalor)
    #
    # Input:
    # - Hp:             prediction horizon
    # - Hw:             where the prediction horizon starts
    # - n_CV:           amount of outputs (CVs)
    # - y_over_bar:     upper limit for output (NB! Static and constant for all MVs)
    # - y_under_bar:    lower limit for output (NB! Static and constant for all MVs)
    #
    # Returns: g vector, shape: ((Hp - Hw + 1) * n_CV * 2) x 1

    one = np.ones((Hp - Hw + 1,))
    gis = []
    for j in range(n_CV):
        factor = np.array([y_over_bar[j], -y_under_bar[j]])
        gis.append(np.kron(one, factor))
    
    return np.reshape(np.hstack(tuple(gis)), (n_CV * (Hp - Hw + 1) * 2, 1)) 

####################################

# -- K matrix -- #
def get_K(Hu, n_MV):
    # Builds the matrix K ((3.6f) and p. 36 in Kufoalor)
    #
    # Inputs:
    # - Hu:     control horizon
    # - n_MV:   amount of inputs (MVs)
    #
    # Returns: K matrix, shape: (Hu * n_MV) x (Hu * n_MV)

    Ki = np.diag(np.ones((Hu,)), k=0) + np.diag(-np.ones((Hu - 1,)), k=-1)
    return np.kron(np.eye(n_MV), Ki)

####################################

# -- Gamma matrix -- #
def get_Gamma(Hu, n_MV):
    # Build the Gamma matrix (p. 36 in Kufoalor)
    #
    # Input:
    # - Hu:
    # - n_MV:
    #
    # Returns: Gamma matrix, shape: (Hu * n_MV) x n_MV

    Gamma_i = np.zeros((Hu, 1))
    Gamma_i[0] = 1
    return np.kron(np.eye(n_MV), Gamma_i)

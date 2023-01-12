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

def get_Theta_ij(Sij, Hp, Hw, Hu, sz):
    # Retrieves elements of S that make up Theta_ij
    #
    # Input:
    # - Sij:  step response coefficients matrix for specific I/O-combination
    # - Hp:   prediction horizon
    # - Hw:   where the prediction horizon starts
    # - Hu:   control horizon
    # - sz:   dimensions of Theta_ij
    #
    # Returns: Theta for output (CV) nr. i and input (MV) nr. j, shape: (Hp - Hw + 1) x Hu

    Theta_ij = np.zeros(sz)
    pad_sz = Hu - Hw
    Sij = np.pad(Sij, pad_sz)
    for j in range(Theta_ij.shape[1]):
        Theta_ij[:, j] = Sij[pad_sz + Hw - j:pad_sz + Hp - j + 1]
    return Theta_ij

def get_Theta(S, Hp, Hw, Hu, n_CV, n_MV):
    # Builds Theta matrix
    #
    # Input:
    # - S:    step response coefficients matrix
    # - Hp:   prediction horizon
    # - Hw:   where the prediction horizon starts
    # - Hu:   control horizon
    # - n_CV: Number of outputs (controlled variables)
    # - n_MV: Number of inputs (manipulated variables)
    #
    # Returns: Theta for the whole MIMO model, shape: (n_CV * (Hp - Hw + 1))x(n_MV * Hu)

    N = S.shape[0] // n_CV
    sz = ((Hp  - Hw + 1), Hu)
    Theta = np.zeros((n_CV * sz[0], n_MV * sz[1])) 
              
    for i in range(n_CV):
        Si = S[i * N:(i + 1) * N]
        for j in range(n_MV):
            Sij = Si[:, j]
            Theta[i * sz[0]:(i + 1) * sz[0], j * sz[1]:(j + 1) * sz[1]] = get_Theta_ij(Sij, Hp, Hw, Hu, sz)

    return Theta

def get_Psi_ij(Sij, Hw, Nj, sz):
    # Retrieves elements of S that make up Psi_ij (p. 37 in Kufoalor)
    #
    # Input:
    # - Sij:  step response coefficients matrix for specific I/O-combination
    # - Hw:   where the prediction horizon starts
    # - Nj:   The largest N among each MV's SISO-models. Note! For S to be whole, every SISO model must be padded, so every Nj == N
    # - sz:   dimensions of Psi_ij
    #
    # Returns: Psi for output (CV) nr. i and input (MV) nr. j, shape: (Hp - Hw + 1) x (Nj - Hw - 1)

    Psi_ij = np.zeros(sz)
    for i in range(Psi_ij.shape[0]):
        for j in range(Psi_ij.shape[1]):
            if (Hw + 1) + j + i < Nj:
                Psi_ij[i, j] = Sij[(Hw + 1) + i + j]
            else:
                Psi_ij[i, j] = Sij[Nj - 1] # To account for zero-indexing

    return Psi_ij

def get_Psi(S, Hp, Hw, Nj, n_CV, n_MV): # TODO: Fix correct dimensions, ref table on p. 41
    # Builds Psi matrix (p.37 in Kufoalor)
    #
    # Input:
    # - S:    step response coefficients matrix
    # - Hp:   prediction horizon
    # - Hw:   where the prediction horizon starts
    # - Nj:   The largest N among each MV's SISO-models. Note! For S to be whole, every SISO model must be padded, so every Nj == N
    # - n_CV: amount of outputs (CVs)
    # - n_MV: amount of inputs (MVs)
    #
    # Returns: Psi matrix, shape: n_CV * (Hp - Hw + 1) x n_MV * (Nj - Hw - 1)
    
    N = S.shape[0] // n_CV
    sz = ((Hp - Hw + 1, Nj - 1))
    Psi = np.zeros((n_CV * sz[0], n_MV * sz[1])) # Since each Nj must be equal, the sum reduces to a multiplication
    for i in range(n_CV):
        Si = S[i * N:(i + 1) * N]
        for j in range(n_MV):
            Sij = Si[:, j]
            Psi[i * sz[0]:(i + 1) * sz[0], j * sz[1]:(j + 1) * sz[1]] = get_Psi_ij(Sij, Hw, Nj, sz)

    return Psi 

def get_Upsilon_ij(Sij, Nj, sz):
    # Retrieves elements of S that make up Upsilon_ij (p. 36 in Kufoalor)
    #
    # Input:
    # - Sij:  step response coefficients matrix
    # - Nj:   The largest N among each MV's SISO-models. Note! For S to be whole, every SISO model must be padded, so every Nj == N
    # - sz:   dimensions of Psi_ij
    #
    # Returns: Upsilon for output (CV) nr. i and input (MV) nr. j, shape: (Hp - Hw + 1) x 1

    return np.array([Sij[Nj - 1] for i in range(sz[0])])

def get_Upsilon(S, Hp, Hw, Nj, n_CV, n_MV):
    # Builds Upsilon matrix (p. 36 in Kufoalor)
    #
    # Input:
    # - S:    step response coefficients matrix
    # - Hp:   prediction horizon
    # - Hw:   where the prediction horizon starts
    # - Nj:   The largest N among each MV's SISO-models. Note! For S to be whole, every SISO model must be padded, so every Nj == N
    # - n_CV: amount of outputs (CVs)
    # - n_MV: amount of inputs (MVs)
    #
    # Returns: Psi matrix, shape: (n_CV * (Hp - Hw + 1)) x n_MV

    N = S.shape[0] // n_CV
    sz = ((Hp - Hw + 1, 1))
    Upsilon = np.zeros((n_CV * sz[0], n_MV * sz[1]))
    for i in range(n_CV):
        Si = S[i * N:(i + 1) * N]
        for j in range(n_MV):
            Sij = Si[:, j]
            Upsilon_ij = get_Upsilon_ij(Sij, Nj, sz)
            Upsilon[i * sz[0]:(i + 1) * sz[0], j] = Upsilon_ij

    return Upsilon

def get_V(y_m, y_hat, Hp, Hw, n_CV):
    # Predicts all future noises based on the constant disturbance model (p. 33 in Kufoalor)
    # 
    # Input:
    # - y_m:    the measured output value at a the current timestep, shape: n_CV x 1
    # - y_hat:  the predicted output value for the current timestep, predicted at the previous timestep, shape: n_CV x 1
    #           Is a vector of shape n_CV for n_CV > 1
    # - n_CV:   amount of outputs (CVs)
    # - Hp:     prediction horizon
    # - Hw:     where the prediction horizon starts
    #
    # Returns: Vector of predicted disturbances for the same prediction horizon as Y(k) uses, shape: n_CV * (Hp - Hw + 1) x 1

    V = np.zeros((n_CV * (Hp - Hw + 1), 1))
    for i in range(n_CV):
        V[i * (Hp - Hw + 1):(i + 1) * (Hp - Hw + 1)] = y_m[i] - y_hat[i]

    return V    

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

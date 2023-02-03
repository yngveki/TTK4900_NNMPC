import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import casadi as ca
import numpy as np
import utils.references as references
from pathlib import Path

# -- Utils -- #
def ReLU(x):
    return x * (x > 0)

def casadi_MLP(layer_szs, weights, biases):
        assert len(layer_szs) >= 3, "Must include at least input layer, hidden layer and output layer!"
        # Defining activation function
        placeholder = ca.MX.sym('placeholder')
        ReLU = ca.Function('ReLU', [placeholder], [placeholder * (placeholder > 0)],
                                    ['relu_in'], ['relu_out'])

        # Setting up neural network
        x0 = ca.MX.sym('x', layer_szs[0])
        x = x0
        for l in range(1, len(layer_szs)):
            W = weights[l-1]
            b = biases[l-1]
            x_t = ca.MX.sym('x', layer_szs[l-1]) # Placeholder for each layer's input

            layer = ca.Function('f_MLP', [x_t], [W.T @ x_t + b],\
                                    ['input'], ['output'])
            x = ReLU(layer(x))

        return ca.Function('f_MLP', [x0], [x], ['input_MLP'], ['output_MLP'])

# -- Proofs of concepts -- #
def basic_example():
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()

    opti.minimize(  (y-x**2)**2   )
    opti.subject_to( x**2+y**2==1 )
    opti.subject_to(       x+y>=1 )

    opti.solver('ipopt')


    sol = opti.solve()

    print(sol.value(x))
    print(sol.value(y))

#! WIP
def update_recursive_cost_constraints_casadi(uk, yk, Y_ref):
    # TODO: How access the variables of Y that are before 0?
    #       -> Maybe define the variables as my+N and mu+N long, and just
    #          initialize all past values to same value as yk?
     
    N = 20
    Q = np.ones((2,2))
    R = np.ones((2,2))
    rho = np.ones((2,))
    mu = 2 # mock value
    my = 2 # mock value

    Y = ca.MX.sym('Y_hat',2,N+1+my)
    DU = ca.MX.sym('DU',2,N+mu)
    epsy = ca.MX.sym('epsy',2,1)

    U = ca.MX.sym('U',2,N)

    # Initialization, because f_MLP needs historical values
    Y[:,0:my] = yk
    Y[:,my] ==   yk    # (1b)
    U[:,0:mu] = uk
    U[:,mu] =   uk

    # Define cost function recursively ("Fold"; see https://web.casadi.org/docs/#for-loop-equivalents)
    cost = 0
    for i in range(N):
        cost += (Y[:,i] - Y_ref[:,i]) @ Q @ (Y[:,i] - Y_ref[:,i]).T + DU[:,i] @ R @ DU[:,i].T

    cost += rho.T @ epsy # This should now be equal to the 'f' keyword taken by the solver
    # cost = ca.Function('cost', [Y,DU,epsy], [cost])

    # Define constraints recursively
    f_MLP = ... # TODO: How implement generic neural network?
    # ca.Function('f_MLP', [y, du, epsy], [cost])

    for i in range(N):
        Y[:,i + 1] == f_MLP([Y[:,i - my:i],U[:,i - 1 - mu:i - 1], U[:,i]])   # (1c)
        U[:,i] == U[:,i - 1] + DU[:,i]      
        
        
def update_recursive_cost_constraints_opti(uk, yk, Y_ref, config, weights, biases):
    # TODO: How access the variables of Y that are before 0?
    #       -> Maybe define the variables as my+N and mu+N long, and just
    #          initialize all past values to same value as yk?
    
    # mock values
    N = config['N']
    mu = config['mu']
    my = config['my']
    Q = config['Q']
    R = config['R']
    rho = config['rho']

    opti = ca.Opti()
    Y = opti.variable(config['n_CV'],N+1+my)
    DU = opti.variable(config['n_MV'],N)
    epsy = opti.variable(2)

    U = opti.variable(2,N+mu)

    # Initialization, because f_MLP needs historical values (opti.set_value()?)
    for i in range(my):
        Y[:,i] = yk
    for i in range(mu):
        U[:,i] = uk
    U[:,mu] =   uk

    # Define cost function recursively ("Fold"; see https://web.casadi.org/docs/#for-loop-equivalents)
    cost = 0
    for i in range(N):
        cost += (Y[:,i] - Y_ref[i]).T @ Q @ (Y[:,i] - Y_ref[i]) + DU[:,i].T @ R @ DU[:,i]

    cost += rho.T @ epsy 
    opti.minimize(cost) # (1a)

    # Define constraints, respecting recursion in (1g)
    constraints = []
    constraints.append(Y[:,my] == yk) # (1b)
    temp = Y[:,my]
    for i in range(N):          
        constraints.append(opti.bounded(config['ylb'] - epsy,\
                                        Y[:,i + 1],\
                                        config['yub'] + epsy)) # (1d)
        constraints.append(opti.bounded(config['dulb'],\
                                        DU[:,i],\
                                        config['duub'])) # (1e)
        constraints.append(opti.bounded(config['ulb'],\
                                        U[:,i],\
                                        config['uub'])) # (1f)
        constraints.append(U[:,i] == U[:,i - 1] + DU[:,i]) # (1g)
    
    constraints.append(opti.bounded(config['elb'],\
                                    epsy,\
                                    config['eub'])) # (1h)

    
    # # Defining the neural network as constraint
    assert len(weights) == len(biases), "There must be a set of biases for each set of weights!"
    x0 = ca.horzcat(U[:,:mu + 1], Y[:,:my + 1])# Input to the network
    x = x0.reshape((x0.shape[0] * x0.shape[1],1))
    for W, b in zip(weights, biases):
        # print(isinstance(x, ca.MX))
        # assert isinstance(x, ca.MX), "x is indeed not symbolic"
        layer = ca.Function('layer', [x], [ca.mtimes(W.T, x) + b],\
                            ['input layer'], ['output layer']) # Defining the current layer
        x = layer([x, W, b])
        relu = ca.Function('ReLU', [x], [x * (x > 0)])
        x = relu(x)
    f_MLP = ca.Function('f_MLP', [x0], [x])

    assert mu == my, "below indexing currently can\'t handle mu != my"
    for i in range(my, N + my):
        #! This indexing goes wrong when mu != my
        constraints.append(Y[:,i + 1] == f_MLP([Y[:,i - my:i],U[:,i - 1 - mu:i - 1], U[:,i]])) # (1c)
        
    opti.subject_to(constraints)

    p_opts = {"expand":True}    
    s_opts = {"max_iter": 100}
    opti.solver('ipopt', p_opts, s_opts)

if __name__ == "__main__":

    N = 20
    n_MV = 2
    n_CV = 2
    mu = 2
    my = 2

    config = {}
    # Horizons
    config['N'] = N
    config['Hu'] = N 
    config['Hp'] = N

    # Weights
    config['Q'] = np.eye(2)
    config['R'] = np.eye(2)
    config['rho'] = 1000 * np.ones((2,1))

    # Constraints
    config['ylb'] = [1,1]
    config['yub'] = [16000,500]
    config['ulb'] = [0,0]
    config['uub'] = [100,10000]
    config['dulb'] = np.array([-0.55,-166.7])
    config['duub'] = np.array([0.55,166.7])
    config['elb'] = [0,0]
    config['eub'] = [1000000,1000000]

    # Timekeeping
    config['delta_t'] = 1
    config['final_t'] = 200
    config['t'] = 0

    # misc parameters
    config['n_MV'] = n_MV
    config['n_CV'] = n_CV
    config['mu'] = mu
    config['my'] = my

    ref_path = Path(__file__).parent / "../../config/refs/testrefs.csv"
    refs = references.ReferenceTimeseries(ref_path, N, config['delta_t'])
    Y_ref = refs.refs_as_lists()
    uk = [50, 5000]     # Fine as temp values?
    yk = [10000, 280]   # Fine as temp values?
    
    # Define a mock net on form 12->10->2 (n_MV * (mu + 1) + n_CV * (my + 1) equals 12)
    weights = []
    weights.append(np.ones((n_MV * (mu + 1) + n_CV * (my + 1), 10)))
    weights.append(np.ones((10,2)))
    biases = []
    biases.append(10)
    biases.append(2)
    update_recursive_cost_constraints_opti(uk, yk, Y_ref, config, weights, biases)

    print("finished :)")
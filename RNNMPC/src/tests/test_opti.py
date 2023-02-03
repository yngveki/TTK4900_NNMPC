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

def build_MLP(weights, biases):
    assert len(weights) == len(biases), "Each set of weights must have a corresponding set of biases!"
    assert len(weights) >= 2, "Must include at least input layer, hidden layer and output layer!"
    
    n_layers = len(weights) # Number of layers (excluding input layer, since no function call happens there)

    placeholder = ca.MX.sym('placeholder')
    ReLU = ca.Function('ReLU', [placeholder], [placeholder * (placeholder > 0)],
                                ['relu_in'], ['relu_out'])

    x_in = ca.MX.sym('x', len(weights[0]))
    x = x_in
    for l, (W, b) in enumerate(zip(weights,biases)):
        layer_sz = len(W)
        x_t = ca.MX.sym('x', layer_sz)
        layer = ca.Function('f_MLP', [x_t], [W.T @ x_t + b],\
                              ['layer_in'], ['layer_out'])
        x = layer(x)
        if (l+1) < n_layers:
            x = ReLU(x)

    return ca.Function('f_MLP', [x_in], [x], ['MLP_in'], ['MLP_out'])

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

    U = opti.variable(2,N+mu) # History does _not_ have +1, since it's mu steps backwards from k; k-1 is part of the mu amount of historical steps

    # Initializing overhead for (1c)
    # Expand so Y and U also contain historical values # TODO: maybe have to use opti.set_value()(?)
    # Note that this has implications for indexing:
    #   -> From (1c): u_{k+i-1:k+i-mu}   -> [mu+i-mu : mu+1+i-1]     -> [i : mu + i] (fra og med k+i-mu, til _og med_ k+i; historie)
    #                 u_{k+i}            -> [mu + 1 + i]             -> [mu + 1 + i] (nåværende)
    #   -> From (1c): y_{k+i:k+i-my}     -> [my+i-my : my+i]         -> [i : my + 1 + i] (fra og med k+i-my, til _og med_ k+i; nåværende pluss historie)
    for i in range(my):
        Y[:,i] = yk
    for i in range(mu):
        U[:,i] = uk
    U[:,mu] =   uk
    

    cost = 0
    for i in range(N):
        cost += (Y[:,my+i] - Y_ref[i]).T @ Q @ (Y[:,my+i] - Y_ref[i]) + DU[:,i].T @ R @ DU[:,i]
    cost += rho.T @ epsy 
    opti.minimize(cost) # (1a)

    # Define constraints, respecting recursion in (1g)
    constraints = []
    f_MLP = build_MLP(weights, biases)

    constraints.append(Y[:,my] == yk) # (1b)
    for i in range(N):   
        # (1c)
        x = ca.horzcat(U[:,i:mu + i + 1], Y[:,i:my + i + 1])
        x = ca.reshape(x, x.numel(), 1)
        constraints.append(Y[:,my + 1 + i] == f_MLP(MLP_in=x)['MLP_out']) 

        # (1d)
        constraints.append(opti.bounded(config['ylb'] - epsy,\
                                        Y[:,my + 1 + i],\
                                        config['yub'] + epsy)) 
                                        
        # (1e)
        constraints.append(opti.bounded(config['dulb'],\
                                        DU[:,i],\
                                        config['duub']))

        # (1f)
        constraints.append(opti.bounded(config['ulb'],\
                                        U[:,mu + i],\
                                        config['uub'])) 

        # (1g)                                        
        constraints.append(U[:,mu + i] == U[:,mu + i - 1] + DU[:,i]) 
    
    # (1h)
    constraints.append(opti.bounded(config['elb'],\
                                    epsy,\
                                    config['eub'])) 

    
    # # Defining the neural network as constraint
    for i in range(N):
        # t1 = U[:,i:mu + i] # slice: u_{k+i-1:k+i-1-mu} -> mu+1+i-1-mu : mu+1+i-1
        # t2 = U[:,mu + 1 + i] # k == mu + 1 (correcting for history)
        # t3 = Y[:,i:my + i + 1] # slice: y_{k+i:k+i-my} -> my+i-my : my+i (correcting for history) (fra og med k+i-my, til og med k+i)
        x = ca.horzcat(U[:,i:mu + i + 1], Y[:,i:my + i + 1])
        x = ca.reshape(x, x.numel(), 1)
        constraints.append(Y[:,my + 1 + i] == f_MLP(MLP_in=x)['MLP_out']) # (1c)
        
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
    in_sz = n_MV * (mu + 1) + n_CV * (my + 1)
    hl_sz = 10
    out_sz = 2
    weights = []
    weights.append(np.ones((in_sz, hl_sz)))
    weights.append(np.ones((hl_sz,out_sz)))
    biases = []
    biases.append(np.ones((hl_sz,)) * 10)
    biases.append(np.ones((out_sz,)) * 2)
    update_recursive_cost_constraints_opti(uk, yk, Y_ref, config, weights, biases)

    print("finished :)")
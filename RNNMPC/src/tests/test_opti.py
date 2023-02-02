import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import casadi as ca
import numpy as np
import utils.references as references
from pathlib import Path

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
        
        
def update_recursive_cost_constraints_opti(uk, yk, Y_ref, config):
    # TODO: How access the variables of Y that are before 0?
    #       -> Maybe define the variables as my+N and mu+N long, and just
    #          initialize all past values to same value as yk?
    
    # mock values
    N = 20
    Q = np.ones((2,2))
    R = np.ones((2,2))
    rho = np.ones((2,1))
    mu = 2 
    my = 2 

    opti = ca.Opti()
    Y = opti.variable(2,N+1+my)
    DU = opti.variable(2,N+mu)
    epsy = opti.variable(2)

    U = opti.variable(2,N)

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

    #! Defined at bottom so debugging of other more trivial stuff can happen above until this finishes
    # TODO: How implement generic neural network?
    # f_MLP = ... 
    # for i in range(N):
    #     opti.subject_to(Y[:,i + 1] == f_MLP([Y[:,i - my:i],U[:,i - 1 - mu:i - 1], U[:,i]])) # (1c)

    opti.subject_to(constraints)

    p_opts = {"expand":True}    
    s_opts = {"max_iter": 100}
    opti.solver('ipopt', p_opts, s_opts)

if __name__ == "__main__":

    N = 20
    config = {}

    # Horizons
    config['Hu'] = N 
    config['Hp'] = N
    # Weights
    config['Q'] = np.eye(2)
    config['P'] = np.eye(2)
    config['rho'] = 1000 * np.ones((2,))

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

    ref_path = Path(__file__).parent / "../../config/refs/testrefs.csv"
    refs = references.ReferenceTimeseries(ref_path, N, config['delta_t'])
    Y_ref = refs.refs_as_lists()
    uk = [50, 5000]     # Fine as temp values?
    yk = [10000, 280]   # Fine as temp values?
    update_recursive_cost_constraints_opti(uk, yk, Y_ref, config)

    print("finished :)")
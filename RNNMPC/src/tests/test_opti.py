import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import casadi as cs
import numpy as np
import utils.references as references
from pathlib import Path

def basic_example():
    opti = cs.Opti()

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

    Y = cs.MX.sym('Y_hat',2,N+1+my)
    DU = cs.MX.sym('DU',2,N+mu)
    epsy = cs.MX.sym('epsy',2,1)

    U = cs.MX.sym('U',2,N)

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
    # cost = cs.Function('cost', [Y,DU,epsy], [cost])

    # Define constraints recursively
    f_MLP = ... # TODO: How implement generic neural network?
    # cs.Function('f_MLP', [y, du, epsy], [cost])

    for i in range(N):
        Y[:,i + 1] == f_MLP([Y[:,i - my:i],U[:,i - 1 - mu:i - 1], U[:,i]])   # (1c)
        U[:,i] == U[:,i - 1] + DU[:,i]      
        
        
def update_recursive_cost_constraints_opti(uk, yk, Y_ref, config):
    # TODO: How access the variables of Y that are before 0?
    #       -> Maybe define the variables as my+N and mu+N long, and just
    #          initialize all past values to same value as yk?
    
    # TODO: Possible to group all constraints together and simply say opti.subject_to(constraints)?
    N = 20
    Q = np.ones((2,2))
    R = np.ones((2,2))
    rho = np.ones((2,))
    mu = 2 # mock value
    my = 2 # mock value

    opti = cs.Opti()
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
        t1 = Y[:,i]
        t2 = Y_ref[i]
        t3 = DU[:,i]
        cost += (Y[:,i] - Y_ref[i]) @ Q @ (Y[:,i] - Y_ref[i]).T + DU[:,i] @ R @ DU[:,i].T

    cost += rho.T @ epsy # This should now be equal to the 'f' keyword taken by the solver
    opti.minimize(cost)

    # Define constraints recursively

    opti.subject_to(Y[:,my] ==   yk)    # (1b)
    for i in range(N):
        opti.subject_to(U[:,i] == U[:,i - 1] + DU[:,i])                                     # (1g)

        opti.subject_to(config['ylb'] - epsy <= Y[:,i + 1] <= config['yub'] + epsy)         # (1d)
        opti.subject_to(config['dulb'] <= DU[:,i] <= config['duub'])                        # (1e)
        opti.subject_to(config['ulb'] <= U[:,i] <= config['uub'])                           # (1f)
    
    opti.subject_to(config['elb'] <= epsy <= config['eub'])                                 # (1h)

    #! Defined at bottom so debugging of other more trivial stuff can happen above until this finishes
    # TODO: How implement generic neural network?
    f_MLP = ... 
    for i in range(N):
        opti.subject_to(Y[:,i + 1] == f_MLP([Y[:,i - my:i],U[:,i - 1 - mu:i - 1], U[:,i]])) # (1c)



if __name__ == "__main__":
    # basic_example()

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
    config['ylb'] = [0,0]
    config['yub'] = [16000,500]
    config['ulb'] = [0,0]
    config['uub'] = [100,10000]
    config['dulb'] = [-0.55,-166.7]
    config['duub'] = [0.55,166.7]
    config['elb'] = [0,0]
    config['eub'] = [1000000,1000000]

    # Timekeeping
    config['delta_t'] = 1
    config['final_t'] = 200
    config['t'] = 0

    ref_path = Path(__file__).parent / "../../config/refs/testrefs.csv"
    refs = references.ReferenceTimeseries(ref_path, N, config['delta_t'])
    Y_ref = refs.ref_series
    uk = [50, 5000]     # Fine as temp values?
    yk = [10000, 280]   # Fine as temp values?
    update_recursive_cost_constraints_opti(uk, yk, Y_ref, config)

    print("finished :)")
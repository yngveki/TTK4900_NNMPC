# Based off of https://web.casadi.org/docs/#document-function

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import casadi as ca
import numpy as np

def basic_example():
    x = ca.MX.sym('x',2)
    y = ca.MX.sym('y')
    f = ca.Function('f', [x,y], [x,np.sin(y)*x],\
                         ['x', 'y'], ['r','q'])
    print(f)

    r0, q0 = f(1.1, 3.3)
    print(f'r0: {r0}')
    print(f'q0: {q0}')

    res = f(x=1.1, y=3.3)
    print('res:', res)

def basic_fold_example(use_fold=False):
    N = 5
    blank = ca.MX.sym('blank')
    f = ca.Function('f', [blank], [blank * 2],\
                        ['f_in'], ['f_out'])
    x0 = ca.MX.sym('x')

    if not use_fold:
        x = x0
        for i in range(N):
            x = f(x)

        F = ca.Function('F',[x0],[x],\
                            ['F_in'], ['F_out'])
    else:
        F = f.fold(N)
    
    print(F)
    res = F(f_in=2)
    print(f'res: {res}')

def iterative_fold_example():
    N = 5
    blank = ca.MX.sym('blank')
    x0 = ca.MX.sym('x')

    x = x0
    for i in range(1, N+1):
        f = ca.Function('f', [blank], [blank + i],\
                    ['f_in'], ['f_out'])
        x = f(x)

    F = ca.Function('F',[x0],[x],\
                        ['F_in'], ['F_out'])
    
    print(F)
    res = F(F_in=0)
    print(f'res: {res}')

# TODO: If possible, would have been smoother using `fold`
def build_nn_ca(weights, biases):
    assert len(weights) == len(biases), "Each set of weights must have a corresponding set of biases!"
    assert len(weights) >= 2, "Must include at least input layer, hidden layer and output layer!"
    
    n_layers = len(weights) # Number of layers (excluding input layer, since no function call happens there)
    # Defining activation function
    placeholder = ca.MX.sym('placeholder')
    ReLU = ca.Function('ReLU', [placeholder], [placeholder * (placeholder > 0)],
                                ['relu_in'], ['relu_out'])

    # Setting up neural network
    x0 = ca.MX.sym('x', len(weights[0]))
    x = x0
    for l, (W, b) in enumerate(zip(weights,biases)):
        layer_sz = len(W)
        x_t = ca.MX.sym('x', layer_sz)
        layer = ca.Function('f_MLP', [x_t], [W.T @ x_t + b],\
                              ['layer_in'], ['layer_out'])
        x = layer(x)
        if (l+1) < n_layers: # Don't want activation function on output layer
            x = ReLU(x)
        # x = ReLU(x)

    return ca.Function('f_MLP', [x0], [x], ['MLP_in'], ['MLP_out'])

# -- RUN -- #
if __name__ == "__main__":
    # basic_fold_example(use_fold=True)
    # iterative_fold_example()

    layer_szs = [2,5,2]
    weights = [np.ones((2,5)), np.ones((5,2))]
    biases = [np.ones((5,)) * 10, np.ones((2,)) * (-3)]
    f_MLP = build_nn_ca(weights, biases)

    res = f_MLP(MLP_in=[1,-10])
    print(f'output from NN: {res}')
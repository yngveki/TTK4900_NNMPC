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


def build_nn_ca(layer_szs):
    assert len(layer_szs) >= 3, "Must include at least input layer, hidden layer and output layer!"
    # Defining activation function
    placeholder = ca.MX.sym('placeholder')
    ReLU = ca.Function('ReLU', [placeholder], [placeholder * (placeholder > 0)],
                                ['relu_in'], ['relu_out'])

    # Setting up neural network
    x0 = ca.MX.sym('x', layer_szs[0])
    x = x0
    for l in range(1, len(layer_szs)):
        W = ca.MX.sym('W', layer_szs[l-1], layer_szs[l])
        b = ca.MX.sym('b', layer_szs[l])

        #! Below line fails after first run-through because x then no
        #! longer is purely symbolic
        layer = ca.Function('f_MLP', [x, W, b], [W.T @ x + b],\
                              ['input', 'weights', 'biases'], ['output'])
        x = ReLU(layer(x, W, b))
        print(x)
    
    # TODO: Try to use "fold" instead, as in `basic_fold_example`
    #! That might not be possible when layers are of different sizes, 
    #! and thus weights and biases must be as well
    f_MLP = ca.Function('F', [x0], [x])
    return f_MLP

# -- RUN -- #
if __name__ == "__main__":
    # basic_fold_example(use_fold=True)
    # iterative_fold_example()

    weights = [np.ones((2,5)), np.ones((5,2))]
    biases = np.array([10, 2], ndmin=2).T
    f_MLP = build_nn_ca([2,10,2])

    print(f'output from NN: {f_MLP(x=[1,2])}')
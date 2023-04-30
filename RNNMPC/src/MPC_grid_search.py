#!/usr/bin/env python3
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import casadi as ca
import numpy as np
from yaml import safe_load

from src.utils.simulate_fmu import init_model, simulate_singlewell_step
from src.neuralnetwork import NeuralNetwork
from src.utils.references import ReferenceTimeseries
from src.utils.custom_timing import Timer
from src.utils.plotting import plot_MPC_step
# import ml_casadi.torch as mc

class RNNMPC:

    def __init__(self, 
                nn_path,
                nn_config_path,
                ref_path,
                mpc_configs):
        """
        Takes in given paths to setup the framework around the OCP
        """
        
        # -- Initialization of variables for later availability -- #
        self.fmu = None

        self.simulated_u = {}
        self.simulated_u['choke'] = []
        self.simulated_u['gas lift'] = []
        self.simulated_u['k'] = 0
        self.simulated_y = {}
        self.simulated_y['gas rate'] = []
        self.simulated_y['oil rate'] = []
        self.simulated_y['k'] = 0
        self.full_refs = {}
        self.full_refs['gas rate'] = []
        self.full_refs['oil rate'] = []
        self.bias = {}
        self.bias['gas rate'] = []
        self.bias['oil rate'] = []
        self.yk = None
        self.yk_hat = [0,0] # Initialize bias to zero
        self.uk = None

        # -- config parameters -- #
        self.config = {}

        # System parameters
        self.config['n_MV'] = mpc_configs['n_MV']
        self.config['n_CV'] = mpc_configs['n_CV']

        # Horizons
        self.config['Hu'] = mpc_configs['Hu']
        self.config['Hp'] = mpc_configs['Hp']

        # Weights
        self.config['Q'] = np.diag(mpc_configs['Q'])
        self.config['R'] = np.diag(mpc_configs['R'])
        self.config['rho'] = mpc_configs['rho']

        # Constraints
        self.config['ylb'] = mpc_configs['ylb']
        self.config['yub'] = mpc_configs['yub']
        self.config['ulb'] = mpc_configs['ulb']
        self.config['uub'] = mpc_configs['uub']
        self.config['dulb'] = mpc_configs['dulb']
        self.config['duub'] = mpc_configs['duub']
        self.config['elb'] = mpc_configs['elb']
        self.config['eub'] = mpc_configs['eub']

        # Solver options
        p_opts = mpc_configs['PLUGIN_OPTIONS']
        s_opts = mpc_configs['SOLVER_OPTIONS']

        # Timekeeping
        self.delta_t = mpc_configs['delta_t']
        self.warm_start_t = mpc_configs['warm_start_t']
        self.final_t = mpc_configs['final_t'] + self.warm_start_t
        self.t = 0

        # -- Set up references -- #
        self.refs = ReferenceTimeseries(ref_path, 
                                        self.config['Hp'], 
                                        self.delta_t,
                                        time=0)

        # -- Load neural network model -- #   
        nn_configs = self._read_yaml(nn_config_path)
        self.config['mu'] = nn_configs['mu']
        self.config['my'] = nn_configs['my']    
        # Load model
        layers = []
        layers.append(self.config['n_MV'] * (self.config['mu'] + 1) + \
                      self.config['n_CV'] * (self.config['my'] + 1))
        self.input_layer = layers[-1]

        layers += nn_configs['hlszs']
        self.hidden_layers = layers[-1]

        layers.append(self.config['n_CV'])
        self.output_layer = layers[-1]

        self.nn = NeuralNetwork(layers=layers, model_path=nn_path)
        self.weights, self.biases = self.nn.extract_coefficients()
        self.f_MLP = self._build_MLP(self.weights, self.biases)

        # -- Set up framework for OCP using Opti from CasADi -- #
        self.opti = ca.Opti()

        self._declare_OCP_variables()

        self.opti.solver('ipopt', p_opts, s_opts)

    def warm_start(self, fmu_path, warm_start_input):
        """
        Simulates the fmu for a few steps to ensure defined state before optimization loop
        """
        # Time is updated after to warm_start_t into self.t
        self.fmu, init_u, init_y, self.t = init_model(fmu_path, 
                                                      self.t, 
                                                      self.final_t, # Needed for initialization, but different from warm start time
                                                      self.delta_t,
                                                      self.warm_start_t,
                                                      vals=warm_start_input)
        
        self.simulated_u['choke'] += init_u[:,0].tolist()
        self.simulated_u['gas lift'] += init_u[:,1].tolist()
        self.simulated_u['k'] = int(self.t // self.delta_t)
        self.simulated_y['gas rate'] += init_y[:,0].tolist()
        self.simulated_y['oil rate'] += init_y[:,1].tolist()
        self.simulated_y['k'] = int(self.t // self.delta_t)

        self.uk = [self.simulated_u['choke'][-1],
                   self.simulated_u['gas lift'][-1]]
        self.yk = [self.simulated_y['gas rate'][-1],
                   self.simulated_y['oil rate'][-1]]
        self.yk_hat = self.yk.copy()
        self.bias['gas rate'].append(self.yk[0] - self.yk_hat[0])
        self.bias['oil rate'].append(self.yk[1] - self.yk_hat[1])

    def _declare_OCP_variables(self):
        self.Hp = self.config['Hp']
        self.Hu = self.config['Hu']
        assert self.Hu == self.Hp, "Given the current OCP-formulation, control and prediction horizons must be the same!"
        self.my = self.config['my']
        self.mu = self.config['mu']
        self.n_slack = len(self.config['rho'])

        self.Y = self.opti.variable(self.config['n_CV'],self.config['Hp']+1+self.my)
        self.V = self.opti.variable(self.config['n_CV'],1) # Assumed constant; no need for full horizon
        self.DU = self.opti.variable(self.config['n_MV'],self.Hu)
        self.epsy = self.opti.variable(self.n_slack)

        self.U = self.opti.variable(self.config['n_MV'],self.Hu+self.mu) # History does _not_ have +1, since it's mu steps backwards from k; k-1 is part of the mu amount of historical steps

    def update_OCP(self):
        self.Y_ref = self.refs.refs_as_lists()

        # (3a)
        cost = 0
        for i in range(self.Hp):
            cost += (self.Y[:,self.my+1+i] - self.Y_ref[i]).T @ self.config['Q'] @ (self.Y[:,self.my+1+i] - self.Y_ref[i]) # TODO: Is this matrix product correct? Isn't Q 2x1?
        for i in range(self.Hu):
            cost += self.DU[:,i].T @ self.config['R'] @ self.DU[:,i] # TODO: Is this matrix product correct? Isn't R 2x1?
        for i in range(self.n_slack):
            cost += self.config['rho'][i] * self.epsy[i]
        self.opti.minimize(cost)

        # Define constraints, respecting recursion in (3c) and (3g)
        constraints = []
        
        # Initializing overhead for (3c)
        # Expand so Y and U also contain historical values
        # Note that this has implications for indexing:
        #   -> From (3c): u_{k+i-1:k+i-mu}   -> [mu+i-mu : mu+1+i-1]     -> [i : mu + i] (fra og med k+i-mu, til _og med_ k+i; historie)
        #                 u_{k+i}            -> [mu + 1 + i]             -> [mu + 1 + i] (nåværende)
        #   -> From (3c): y_{k+i:k+i-my}     -> [my+i-my : my+i]         -> [i : my + 1 + i] (fra og med k+i-my, til _og med_ k+i; nåværende pluss historie)
        if self.yk is None:
            self.yk = [0,0]
            print(f'yk was not defined during warm start, and was now set to {self.yk}')
        if self.uk is None:
            self.uk = [10,2000]
            print(f'uk was not defined during warm start, and was now set to {self.uk}')
        
        # Updating to correct past values
        for i in range(self.my):
            self.Y[:,i] = self.yk
        for i in range(self.mu):
            self.U[:,i] = self.uk
        
        # (3b)
        constraints.append(self.Y[:,self.my] == self.yk) 
        
        l_U = self.U.shape[1]
        l_Y = self.Y.shape[1]
        for i in range(self.Hp):
               
            # (3c)
            # Since we want to index from current->past, we stride negatively. This makes indexing tricksy. Bear with me:
            # U_(k) -> U_(k-m_u) = U[-len(U) + (mu + 1) - 1:-l - 1:-1] = U[-l + mu:-l - 1:-1]
            # Which generalizes to
            # U_(k+i) -> U_(k-m_u+i) = U[-l + mu + i:-l - 1 + i:-1]
            # 
            # Exactly the same applies to Y.
            x = ca.horzcat(self.U[0,-l_U + self.mu + i:-l_U - 1 + i:-1],
                           self.U[1,-l_U + self.mu + i:-l_U - 1 + i:-1],
                           self.Y[0,-l_Y + self.mu + i:-l_Y - 1 + i:-1],
                           self.Y[1,-l_Y + self.mu + i:-l_Y - 1 + i:-1])
            constraints.append(self.Y[:,self.my + 1 + i] == self.f_MLP(MLP_in=x)['MLP_out'] + self.V) 
        
            # (3d)
            constraints.append(self.opti.bounded(self.config['ylb'] - self.epsy,\
                                            self.Y[:,self.my + 1 + i],\
                                            self.config['yub'] + self.epsy)) 
        for i in range(self.Hu):
            # (3e)
            constraints.append(self.opti.bounded(self.config['dulb'],\
                                            self.DU[:,i],\
                                            self.config['duub']))

            # (3f)
            constraints.append(self.opti.bounded(self.config['ulb'],\
                                            self.U[:,self.mu + i],\
                                            self.config['uub'])) 

            # (3g)                                        
            constraints.append(self.U[:,self.mu + i] == self.U[:,self.mu + i - 1] + self.DU[:,i]) 
        
        # (3h)
        self.V[0] = self.bias['gas rate'][-1]
        self.V[1] = self.bias['oil rate'][-1]

        # (3i)
        constraints.append(self.epsy >= self.config['elb']) # Don't need upper bound

        self.opti.subject_to() # Reset constraints to avoid additivity
        self.opti.subject_to(constraints)

    def solve_OCP(self, debug=False, plot=False):
        sol = self.opti.solve() #! Takes a very long time before even starting to iterate - some sort of initialization? - probably normal, though
        self.uk = sol.value(self.U)[:,self.mu]

        # For debugging purposes
        if debug:
            t1 = sol.value(self.Y)
            t2 = sol.value(self.DU)
            t3 = sol.value(self.U)
            print('debugging')

        if plot:
            # TODO: make plot of step (du vet det derre helt standard MPC-plottet)
            print('Plots per step not yet implemented')
            return NotImplementedError

    def iterate_system(self):
        gas_rate_k, oil_rate_k, \
        choke_act_k, gas_lift_act_k, \
        _, _ = simulate_singlewell_step(self.fmu, 
                                        self.t, 
                                        self.delta_t, 
                                        self.uk) # measurement from FMU, i.e. result from previous actuation

        self.yk = [gas_rate_k, oil_rate_k]

        # Append puts current at end of array
        self.simulated_u['choke'].append(choke_act_k)
        self.simulated_u['gas lift'].append(gas_lift_act_k)
        self.simulated_u['k'] += 1
        self.simulated_y['gas rate'].append(gas_rate_k)
        self.simulated_y['oil rate'].append(oil_rate_k)
        self.simulated_y['k'] += 1 
        self.full_refs['gas rate'].append(self.Y_ref[0][0])
        self.full_refs['oil rate'].append(self.Y_ref[0][1])
        
        x = []
        x.extend(self.simulated_u['choke'][-1:-self.mu-2:-1])      # current->past (want 1 + my (current _and_ past), hence -2)
        x.extend(self.simulated_u['gas lift'][-1:-self.mu-2:-1])   # current->past (want 1 + my (current _and_ past), hence -2)
        x.extend(self.simulated_y['gas rate'][-1:-self.my-2:-1])   # current->past (want 1 + my (current _and_ past), hence -2)
        x.extend(self.simulated_y['oil rate'][-1:-self.my-2:-1])   # current->past (want 1 + my (current _and_ past), hence -2)
        self.yk_hat = self.f_MLP(MLP_in=x)['MLP_out']
        self.bias['gas rate'].append(self.yk[0] - float(self.yk_hat[0]))
        self.bias['oil rate'].append(self.yk[1] - float(self.yk_hat[1]))

        self.t += self.delta_t

    def save_data(self, data_path):
        np.save(data_path / 't.npy', np.linspace(0, self.final_t, num=self.final_t // self.delta_t))
        np.save(data_path / 'gas_rate.npy', self.simulated_y['gas rate'])
        np.save(data_path / 'oil_rate.npy', self.simulated_y['oil rate'])
        np.save(data_path / 'choke.npy', self.simulated_u['choke'])
        np.save(data_path / 'gas_lift.npy', self.simulated_u['gas lift'])
    
    # --- Private funcs --- #
    def _read_yaml(self, path):
        """
        Returns content of a YAML-file at given path
        """
        
        with open(path, "r") as f:
            return safe_load(f)

    def _build_MLP(self, weights, biases):
        assert len(weights) == len(biases), "Each set of weights must have a corresponding set of biases!"
        assert len(weights) >= 2, "Must include at least input layer, hidden layer and output layer!"
        
        n_layers = len(weights) # Number of layers (excluding input layer, since no function call happens there)

        placeholder = ca.MX.sym('placeholder')
        # ReLU = ca.Function('ReLU', [placeholder], [placeholder * (placeholder > 0)],
        #                             ['relu_in'], ['relu_out'])
        # TODO: More robust fetch of leak_rate (currently just hardcoded into neuralnetwork.py)
        leak_rate = 0.2
        LReLU = ca.Function('LReLU', [placeholder], [ca.if_else(placeholder >= 0, placeholder, leak_rate * placeholder)],
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
                x = LReLU(x)

        return ca.Function('f_MLP', [x_in], [x], ['MLP_in'], ['MLP_out'])
        
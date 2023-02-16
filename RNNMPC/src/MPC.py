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
# import ml_casadi.torch as mc

class RNNMPC:

    def __init__(self, 
                nn_path,
                mpc_config_path,
                nn_config_path,
                ref_path):
        """
        Takes in given paths to setup the framework around the OCP
        """
        
        # -- Initialization of variables for later availability -- #
        self.fmu = None

        self.simulated_u = {}
        self.simulated_u['init'] = {'choke': [], 'gas lift': []} # Will keep data from warm_start
        self.simulated_u['sim'] = {'choke': [], 'gas lift': []}  # Will keep data from control loop
        self.simulated_u['full'] = {'choke': [], 'gas lift': []} # Concatenates the two above
        self.simulated_y = {}
        self.simulated_y['init'] = {'gas rate': [], 'oil rate': []} # Will keep data from warm_start
        self.simulated_y['sim'] = {'gas rate': [], 'oil rate': []}  # Will keep data from control loop
        self.simulated_y['full'] = {'gas rate': [], 'oil rate': []} # Concatenates the two above
        self.full_refs = {}
        self.full_refs['gas rate'] = []
        self.full_refs['oil rate'] = []
        self.yk = None
        self.uk = None

        # -- config parameters -- #
        configs = self._read_yaml(mpc_config_path)
        self.config = {}

        # System parameters
        self.config['n_MV'] = configs['SYSTEM_PARAMETERS']['n_MV']
        self.config['n_CV'] = configs['SYSTEM_PARAMETERS']['n_CV']

        # Horizons
        self.config['Hu'] = configs['TUNING_PARAMETERS']['Hu']
        self.config['Hp'] = configs['TUNING_PARAMETERS']['Hp']

        # Weights
        self.config['Q'] = np.diag(configs['TUNING_PARAMETERS']['Q'])
        self.config['R'] = np.diag(configs['TUNING_PARAMETERS']['R'])
        self.config['rho'] = configs['TUNING_PARAMETERS']['rho']

        # Constraints
        self.config['ylb'] = configs['TUNING_PARAMETERS']['ylb']
        self.config['yub'] = configs['TUNING_PARAMETERS']['yub']
        self.config['ulb'] = configs['TUNING_PARAMETERS']['ulb']
        self.config['uub'] = configs['TUNING_PARAMETERS']['uub']
        self.config['dulb'] = configs['TUNING_PARAMETERS']['dulb']
        self.config['duub'] = configs['TUNING_PARAMETERS']['duub']
        self.config['elb'] = configs['TUNING_PARAMETERS']['elb']
        self.config['eub'] = configs['TUNING_PARAMETERS']['eub']

        # Solver parameters
        self.config['expand'] = configs['SOLVER_PARAMETERS']['expand']
        self.config['max_iter'] = configs['SOLVER_PARAMETERS']['max_iter']

        # Timekeeping
        self.delta_t = configs['RUNNING_PARAMETERS']['delta_t']
        self.final_t = configs['RUNNING_PARAMETERS']['final_t']
        self.t = 0

        # -- Set up references -- #
        self.refs = ReferenceTimeseries(ref_path, 
                                        self.config['Hp'], 
                                        self.delta_t,
                                        time=0)

        # -- Load neural network model -- #   
        configs = self._read_yaml(nn_config_path)
        self.config['mu'] = configs['STRUCTURE']['mu']
        self.config['my'] = configs['STRUCTURE']['my']    
        # Load model
        layers = []
        layers.append(self.config['n_MV'] * (self.config['mu'] + 1) + \
                      self.config['n_CV'] * (self.config['my'] + 1))
        self.input_layer = layers[-1]

        layers += configs['STRUCTURE']['hlszs']
        self.hidden_layers = layers[-1]

        layers.append(self.config['n_CV'])
        self.output_layer = layers[-1]

        self.nn = NeuralNetwork(layers=layers, model_path=nn_path)
        self.weights, self.biases = self.nn.extract_coefficients()
        self.f_MLP = self._build_MLP(self.weights, self.biases)
        # self.f_MLP = self._build_MLP(layers, nn_path)

        # -- Set up framework for OCP using Opti from CasADi -- #
        self.opti = ca.Opti()

        self.update_OCP()

        p_opts = {"expand":self.config['expand']}    
        s_opts = {"max_iter": self.config['max_iter']}
        self.opti.solver('ipopt', p_opts, s_opts)

    def warm_start(self, fmu_path, warm_start_t=1000):
        """
        Simulates the fmu for a few steps to ensure defined state before optimization loop
        """
        self.fmu, init_u, init_y = init_model(fmu_path, 
                                                start_time=self.t, 
                                                final_time=self.final_t, # Needed for initialization, but different from warm start time
                                                delta_t=self.delta_t,
                                                warm_start_t=warm_start_t)
        
        self.simulated_u['init']['choke'] = init_u[:,0].tolist()
        self.simulated_u['init']['gas lift'] = init_u[:,1].tolist()
        self.simulated_y['init']['gas rate'] = init_y[:,0].tolist()
        self.simulated_y['init']['oil rate'] = init_y[:,1].tolist()

    def update_OCP(self):
        Hp = self.config['Hp']
        Hu = self.config['Hu']
        my = self.config['my']
        mu = self.config['mu']
        n_slack = len(self.config['rho'])

        Y = self.opti.variable(self.config['n_CV'],self.config['Hp']+1+my)
        DU = self.opti.variable(self.config['n_MV'],Hu)
        epsy = self.opti.variable(n_slack)

        self.U = self.opti.variable(self.config['n_MV'],Hu+mu) # History does _not_ have +1, since it's mu steps backwards from k; k-1 is part of the mu amount of historical steps
        self.Y_ref = self.refs.refs_as_lists()

        # (1a)
        cost = 0
        for i in range(Hp):
            cost += (Y[:,my+i] - self.Y_ref[i]).T @ self.config['Q'] @ (Y[:,my+i] - self.Y_ref[i])
        for i in range(Hu):
            cost += DU[:,i].T @ self.config['R'] @ DU[:,i]
        for i in range(n_slack):
            cost += self.config['rho'][i] * epsy[i]
        self.opti.minimize(cost)

        # Define constraints, respecting recursion in (1c) and (1g)
        constraints = []
        
        # Initializing overhead for (1c)
        # Expand so Y and U also contain historical values
        # Note that this has implications for indexing:
        #   -> From (1c): u_{k+i-1:k+i-mu}   -> [mu+i-mu : mu+1+i-1]     -> [i : mu + i] (fra og med k+i-mu, til _og med_ k+i; historie)
        #                 u_{k+i}            -> [mu + 1 + i]             -> [mu + 1 + i] (nåværende)
        #   -> From (1c): y_{k+i:k+i-my}     -> [my+i-my : my+i]         -> [i : my + 1 + i] (fra og med k+i-my, til _og med_ k+i; nåværende pluss historie)
        if self.yk is None:
            self.yk = [0,0] # TODO: Valid start value?
        if self.uk is None:
            self.uk = [10,2000] # TODO: Valid start value?
        
        for i in range(my):
            Y[:,i] = self.yk
        for i in range(mu):
            self.U[:,i] = self.uk
        self.U[:,mu] = self.uk

        """
        constraints.append(Y[:,my] == yk) # (1b)
        """
        for i in range(Hp):   
            """
            # (1c)
            x = ca.horzcat(U[:,i:mu + i + 1], Y[:,i:my + i + 1])
            x = ca.reshape(x, x.numel(), 1)
            constraints.append(Y[:,my + 1 + i] == self.f_MLP(MLP_in=x)['MLP_out']) 

            # (1d)
            constraints.append(self.opti.bounded(self.config['ylb'] - epsy,\
                                            Y[:,my + 1 + i],\
                                            self.config['yub'] + epsy)) 

        for i in range(Hu):
            # (1e)
            constraints.append(self.opti.bounded(self.config['dulb'],\
                                            DU[:,i],\
                                            self.config['duub']))
        """

            # (1f)
            constraints.append(self.opti.bounded(self.config['ulb'],\
                                            self.U[:,mu + i],\
                                            self.config['uub'])) 

            # (1g)                                        
            constraints.append(self.U[:,mu + i] == self.U[:,mu + i - 1] + DU[:,i]) 
        
        # (1h)
        constraints.append(epsy >= self.config['elb']) # Don't need upper bound
        # constraints.append(self.opti.bounded(self.config['elb'],\
        #                                 epsy,\
        #                                 self.config['eub']))

        self.opti.subject_to(constraints) 

    def solve_OCP(self):
        #! Jacobians become bigger and bigger, and solving takes longer and longer!
        # TODO: Should I provide initial state for solver? (opti.set_initial(<variable_name>, <value>))
        sol = self.opti.solve() # Takes a very long time before even starting to iterate - some sort of initialization?
        self.uk = sol.value(self.U)[:,0]     

    def iterate_system(self):
        gas_rate_k, oil_rate_k, \
        choke_act_k, gas_lift_act_k, \
        _, _ = simulate_singlewell_step(self.fmu, 
                                        self.t, 
                                        self.final_t, 
                                        self.uk) # measurement from FMU, i.e. result from previous actuation

        self.yk = [gas_rate_k, oil_rate_k]
        self.simulated_u['sim']['choke'].append(choke_act_k)
        self.simulated_u['sim']['gas lift'].append(gas_lift_act_k)
        self.simulated_y['sim']['gas rate'].append(gas_rate_k)
        self.simulated_y['sim']['oil rate'].append(oil_rate_k)
        self.full_refs['gas rate'].append(self.Y_ref[0][0])
        self.full_refs['oil rate'].append(self.Y_ref[0][1])
        
        self.t += self.delta_t

    def merge_sim_data(self):
        """
        Convenience function to make the full timeseries more convenient
        """
        self.simulated_y['full']['gas rate'] = self.simulated_y['init']['gas rate'] + self.simulated_y['sim']['gas rate']
        self.simulated_y['full']['oil rate'] = self.simulated_y['init']['oil rate'] + self.simulated_y['sim']['oil rate']
        self.simulated_u['full']['choke'] = self.simulated_u['init']['choke'] + self.simulated_u['sim']['choke']
        self.simulated_u['full']['gas lift'] = self.simulated_u['init']['gas lift'] + self.simulated_u['sim']['gas lift']

    def save_data(self, data_path):
        np.save(data_path / 't.npy', np.linspace(0, self.final_t, num=self.final_t // self.delta_t))
        np.save(data_path / 'gas_rate.npy', self.simulated_y['full']['gas rate'])
        np.save(data_path / 'oil_rate.npy', self.simulated_y['full']['oil rate'])
        np.save(data_path / 'choke.npy', self.simulated_u['full']['choke'])
        np.save(data_path / 'gas_lift.npy', self.simulated_u['full']['gas lift'])
    
    # --- Private funcs --- #
    def _read_yaml(self, file_path):
        """
        Returns content of a YAML-file at given path
        """
        
        with open(file_path, "r") as f:
            return safe_load(f)

    # def _build_MLP(self, layers, path):
    #     nn = mc.nn.CasadiNeuralNetwork(layers)
    #     nn.load(path)

    #     casadi_sym_inp = ca.MX.sym('inp', layers[0])
    #     casadi_sym_out = nn(casadi_sym_inp)
    #     return ca.Function('model2',
    #                             [casadi_sym_inp],
    #                             [casadi_sym_out],
    #                             ['MLP_in'],
    #                             ['MLP_out'])

    def _build_MLP(self, weights, biases):
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
        
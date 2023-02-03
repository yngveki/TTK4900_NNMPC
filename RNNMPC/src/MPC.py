#!/usr/bin/env python3

import casadi as ca
from yaml import safe_load

from utils.simulate_fmu import init_model, simulate_singlewell_step
from neuralnetwork import NeuralNetwork
from utils.references import ReferenceTimeseries

class RNNMPC:

    def __init__(self, 
                nn_path,
                config_path,
                ref_path):
        """
        Takes in given paths to setup the framework around the OCP
        """
        
        # -- Initialization of variables for later availability -- #
        self.fmu = None

        self.simulated_u = {}
        self.simulated_u['init'] = [] # Will keep data from warm_start
        self.simulated_u['sim'] = []  # Will keep data from control loop
        self.simulated_u['full'] = [] # Concatenates the two above
        self.simulated_y = {}
        self.simulated_y['init'] = [] # Will keep data from warm_start
        self.simulated_y['sim'] = []  # Will keep data from control loop
        self.simulated_y['full'] = [] # Concatenates the two above

        self.yk = None
        self.uk = None

        # -- config parameters -- #
        configs = self._read_yaml(config_path)
        self.config = {}

        # System parameters
        self.config['n_MV'] = configs['SYSTEM_PARAMETERS']['n_MV']
        self.config['n_CV'] = configs['SYSTEM_PARAMETERS']['n_CV']
        self.config['mu'] = configs['SYSTEM_PARAMETERS']['mu']
        self.config['my'] = configs['SYSTEM_PARAMETERS']['my']

        # Horizons
        self.config['Hu'] = configs['TUNING_PARAMETERS']['Hu']
        self.config['Hp'] = configs['TUNING_PARAMETERS']['Hp']

        # Weights
        self.config['Q'] = configs['TUNING_PARAMETERS']['Q']
        self.config['P'] = configs['TUNING_PARAMETERS']['P']
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

        # Timekeeping
        self.config['delta_t'] = configs['RUNNING_PARAMETERS']['delta_t']
        self.config['final_t'] = configs['RUNNING_PARAMETERS']['final_t']
        self.config['t'] = 0

        # -- Set up references -- #
        self.refs = ReferenceTimeseries(ref_path, 
                                        self.config['Hp'], 
                                        self.config['delta_t'],
                                        time=0)

        # -- Load neural network model -- #        
        # Load model
        self.model = NeuralNetwork()
        self.model.load(nn_path)

        # Extract weights and biases # TODO: Verify that they are lists of ndarrays
        self.weights, self.biases = self.model.extract_coefficients()

        # -- Set up framework for OCP using Opti from CasADi -- #
        self.opti = ca.Opti()

        self.update_OCP()

        p_opts = {"expand":self.config['SOLVER_PARAMETERS']['expand']}    
        s_opts = {"max_iter": self.config['SOLVER_PARAMETERS']['max_iter']}
        self.opti.solver('ipopt', p_opts, s_opts)

    def warm_start(self, fmu_path, warm_start_t=1000):
        """
        Simulates the fmu for a few steps to ensure defined state before optimization loop
        """
        self.fmu, \
        self.simulated_u['init'], \
        self.simulated_y['init'] = init_model(fmu_path, 
                                            start_time = self.config['t'], 
                                            final_time = self.config['final_t'], # Needed for initialization, but different from warm start time
                                            delta_t = self.config['delta_t'],
                                            warm_start_t=2000) # TODO: Figure out adequate value

    def update_OCP(self):
        Hp = self.config['Hp']
        Hu = self.config['Hu']
        my = self.config['my']
        mu = self.config['mu']

        Y = self.opti.variable(self.config['n_CV'],self.config['Hp']+1+my)
        DU = self.opti.variable(self.config['n_MV'],Hu)
        epsy = self.opti.variable(2)

        U = self.opti.variable(2,Hu+mu) # History does _not_ have +1, since it's mu steps backwards from k; k-1 is part of the mu amount of historical steps
        Y_ref = self.refs.refs_as_lists()

        # (1a)
        cost = 0
        for i in range(Hp):
            cost += (Y[:,my+i] - Y_ref[i]).T @ \
                    self.config['Q'] @ \
                    (Y[:,my+i] - Y_ref[i])
        for i in range(Hu):
            DU[:,i].T @ self.config['R'] @ DU[:,i]
        cost += self.config['rho'].T @ epsy 
        self.opti.minimize(cost)

        # Define constraints, respecting recursion in (1c) and (1g)
        constraints = []
        f_MLP = self._build_MLP(self.weights, self.biases)

        constraints.append(Y[:,my] == yk) # (1b)

        # Initializing overhead for (1c)
        # Expand so Y and U also contain historical values
        # Note that this has implications for indexing:
        #   -> From (1c): u_{k+i-1:k+i-mu}   -> [mu+i-mu : mu+1+i-1]     -> [i : mu + i] (fra og med k+i-mu, til _og med_ k+i; historie)
        #                 u_{k+i}            -> [mu + 1 + i]             -> [mu + 1 + i] (nåværende)
        #   -> From (1c): y_{k+i:k+i-my}     -> [my+i-my : my+i]         -> [i : my + 1 + i] (fra og med k+i-my, til _og med_ k+i; nåværende pluss historie)
        if self.yk is None:
            yk = 0
        if self.uk is None:
            uk = 0
        
        for i in range(my):
            Y[:,i] = yk
        for i in range(mu):
            U[:,i] = uk
        U[:,mu] = uk

        for i in range(Hp):   
            # (1c)
            x = ca.horzcat(U[:,i:mu + i + 1], Y[:,i:my + i + 1])
            x = ca.reshape(x, x.numel(), 1)
            constraints.append(Y[:,my + 1 + i] == f_MLP(MLP_in=x)['MLP_out']) 

            # (1d)
            constraints.append(self.opti.bounded(self.config['ylb'] - epsy,\
                                            Y[:,my + 1 + i],\
                                            self.config['yub'] + epsy)) 

        for i in range(Hu):
            # (1e)
            constraints.append(self.opti.bounded(self.config['dulb'],\
                                            DU[:,i],\
                                            self.config['duub']))

            # (1f)
            constraints.append(self.opti.bounded(self.config['ulb'],\
                                            U[:,mu + i],\
                                            self.config['uub'])) 

            # (1g)                                        
            constraints.append(U[:,mu + i] == U[:,mu + i - 1] + DU[:,i]) 
        
        # (1h)
        constraints.append(self.opti.bounded(self.config['elb'],\
                                        epsy,\
                                        self.config['eub'])) 
            
        self.opti.subject_to(constraints)

    def solve_OCP(self):
        sol = self.opti.solve()
        res = sol.value(self.opti.x).full() # How retrieve full vector of values?

        self.uk = ... # TODO

    def iterate_system(self):
        gas_rate_k, oil_rate_k, \
        choke_act_k, gas_lift_act_k, \
        _, _ = simulate_singlewell_step(self.model, 
                                        self.config['t'], 
                                        self.config['final_t'], 
                                        self.uk) # measurement from FMU, i.e. result from previous actuation

        self.yk = [gas_rate_k, oil_rate_k]
        self.simulated_y['sim'].append([gas_rate_k, oil_rate_k])
        self.simulated_u['sim'].append([choke_act_k, gas_lift_act_k])   

    def merge_sim_data(self):
        """
        Convenience function to make the full timeseries more convenient
        """
        # TODO: If this is done at init, will the ['full'] array update dynamically? -> TEST
        self.simulated_u['full'] = self.simulated_u['init'] + self.simulated_u['sim']
        self.simulated_y['full'] = self.simulated_y['init'] + self.simulated_y['sim']                                                                        
    
    # --- Private funcs --- #
    def _read_yaml(self, file_path):
        """
        Returns content of a YAML-file at given path
        """
        
        with open(file_path, "r") as f:
            return safe_load(f)

    def _build_MLP(weights, biases):
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
        
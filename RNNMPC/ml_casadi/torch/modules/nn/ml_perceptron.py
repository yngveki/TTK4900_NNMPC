import torch
from ml_casadi.torch.modules import TorchMLCasadiModule
from ml_casadi.torch.modules.nn import Linear
from ml_casadi.torch.modules.nn import activation as activations

from ml_casadi.common import MLCasadiModule


class MultiLayerPerceptron(TorchMLCasadiModule):
    def __init__(self, input_size, hidden_size, output_size, n_hidden, activation=None):
        super().__init__()
        assert n_hidden >= 1, 'There must be at least one hidden layer'
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = Linear(input_size, hidden_size)

        hidden = []
        for i in range(n_hidden-1):
            hidden.append((Linear(hidden_size, hidden_size)))
        self.hidden_layers = torch.nn.ModuleList(hidden)

        self.output_layer = Linear(hidden_size, output_size)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            self.act = getattr(activations, activation)()
        else:
            self.act = activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        y = self.output_layer(x)
        return y

class CasadiNeuralNetwork(MLCasadiModule, torch.nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert layers != None, "You need to provide sizes for the layers!"
        assert len(layers) >= 3, "You need to define at least an input, hidden and output layer"
        self.layers = layers

        temp = []
        for idx in range(len(layers) - 1):
            temp.append(Linear(layers[idx], layers[idx + 1]))

        self.layers = torch.nn.ModuleList(temp)

        self.act = getattr(activations, 'ReLU')()

    def forward(self, x):
        """
        Performs a forward pass of the input data x and returns the output.
        
        Args:
            :param x: the given input data to be fed through the neural network.
        """
        input = x
        for l in self.layers[:-1]:
            input = self.act(l(input))

        return self.layers[-1](input)
        

    def load(self, path):
        """
        Loads the model at the given path into the current instance of self.

        Args:
            :param path: the given path from which to load the model.
        """
        self.load_state_dict(torch.load(path))

        # Default to evaluation mode when loaded
        self.eval()

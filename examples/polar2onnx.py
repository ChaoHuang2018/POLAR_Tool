import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnitTest:
    deltaT = 0.2
    maxIteration = 35
    def __init__(self):
        x = np.random.uniform(0.8, 0.9)
        y = np.random.uniform(0.5, 0.6)
        self.state = np.array([x, y])
        self.t = 0
    
    def _f(self, state, u):
        f = np.zeros(len(state))
        x, y = state
        f[0] = y
        f[1] = u*y**2 - x
        return f

    def next(self, x, u):
        simulation_step = 0.01
        steps = int(self.deltaT / simulation_step)
        for _ in range(steps):
            x = x + simulation_step* (self._f(x, u))
        return x

    def step(self, u):
        self.state = self.next(self.state, u)
        self.t += 1
        return self.state, self.t == self.maxIteration
        

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, control_size, hidden_size_list, activation_list):
        """
        Init the neural network with layers and activations information
        """
        super(NeuralNetwork, self).__init__()
        assert len(hidden_size_list) + 1 == len(activation_list)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_size_list[0]))
        for i in range(len(hidden_size_list)-1):
            self.layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
        self.layers.append(nn.Linear(hidden_size_list[-1], control_size))
        print(self.layers)
        # assert False
        self.activations = []
        # self.affine = nn.Linear(control_size, control_size)

        for act in activation_list:
            if act == "ReLU":
                self.activations.append(nn.ReLU())
            elif act == "sigmoid":
                self.activations.append(nn.Sigmoid())
            elif act == "tanh":
                self.activations.append(nn.Tanh())
            elif act == "Affine":
                self.activations.append('Affine')
            else:
                raise NotImplementedError
    
    def forward(self, state):
        assert len(self.layers) == len(self.activations) 
        x = state
        for i in range(len(self.layers)):
            if self.activations[i] == 'Affine':
                x = self.layers[i](x)
            else:
                x = self.activations[i](self.layers[i](x))
        return x

def load_weights_bias(file):
    """
    input: polar neural network controller txt file
    output: transformed onnx NN model
    """
    with open(file) as inputfile:
        lines = inputfile.readlines()
    params = []
    activations = []
    for i, text in enumerate(lines):
        try:
            params.append(eval(text))
        except:
            activations.append(text[:-1])
    state_size, control_size, hidden_size = params[0], params[1], params[2]
    offset, scale = params[-2], params[-1]
    hidden_size_list = params[3: 3+hidden_size]
    nn_params = params[3+hidden_size:-2]
    nn_structure_list = []
    nn_structure_list.append(state_size)
    nn_structure_list += hidden_size_list
    nn_structure_list.append(control_size)
    pointer = 0
    torch_param_list = []
    
    for i in range(len(nn_structure_list) - 1):
        weights = np.zeros((nn_structure_list[i+1], nn_structure_list[i]))
        bias = np.zeros(nn_structure_list[i+1])

        for j in range(nn_structure_list[i+1]):
            for k in range(nn_structure_list[i]):
                weights[j][k] = nn_params[pointer]
                pointer += 1
            bias[j] = nn_params[pointer]
            pointer += 1
        torch_param_list.append(weights)
        torch_param_list.append(bias)
        
    torch_param_list.append(np.array([[scale]]))
    torch_param_list.append(np.array([-offset * scale]))
    NN = NeuralNetwork(state_size, control_size, hidden_size_list, activations)
    print(NN)

    idx = 0
    for name, layer_param in NN.named_parameters():
        print(name, layer_param.shape, torch_param_list[idx].shape)
        layer_param.data = nn.parameter.Parameter(torch.from_numpy(torch_param_list[idx]).float())
        idx += 1
    x = torch.randn(1, state_size, requires_grad=True)
    torch.onnx.export(NN, x, file+".onnx")
    return NN
    



if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    NN = load_weights_bias(filename)
